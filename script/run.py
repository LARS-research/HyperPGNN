import os
import sys
import math
import pprint

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util

separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, test_data=None):
    # here I use the global variable writer, so that I can log the result of valid to tensorboard

    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    # train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    train_triplets = train_data.statements
    maxlen = train_data.maxlen
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    # build a dict from the triplets, the dict is used to compute the negative
    if cfg.task.n_ary_negative:
        gt_triplets = train_data.gt_statements
        hr2t_dict = defaultdict(list)
        tr2h_dict = defaultdict(list)
        for i in range(gt_triplets.shape[0]):
            statement_i = gt_triplets[i, :]
            h, t, r, quals = statement_i[0], statement_i[1], statement_i[2], statement_i[3:]
            hr2t_dict[(int(h), int(r), *quals.tolist())] = int(t)
            tr2h_dict[(int(t), int(r), *quals.tolist())] = int(h)
    else:
        hr2t_dict = None
        tr2h_dict = None
      

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    start_epoch = 0
    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        try:
            start_epoch = state["epoch"]
        except:
            print("No epoch find in checkpoint")
            start_epoch = 0
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    # step = math.ceil(cfg.train.num_epoch / 10)
    if "step" in cfg.train:
        step = cfg.train.step
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(start_epoch, cfg.train.num_epoch, step):
        parallel_model.train()
        # train for #step epochs and then eval on val/test dataset
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in train_loader:
                assert batch.shape[1] == maxlen
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative, hr2t_dict=hr2t_dict, tr2h_dict=tr2h_dict)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)
                writer.add_scalar("avg_loss", avg_loss, epoch)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data, global_step=epoch, split="valid")
        if test_data is not None:
            if rank == 0:
                logger.warning(separator)
                logger.warning("Evaluate on test")
            result = test(cfg, model, test_data, filtered_data=filtered_data, global_step=epoch, split="test")
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None, global_step=None, split=None):
    # here I use the global variable writer, so that I can log the result of valid to tensorboard
    # Liushuzhi add global_step. Both valid and test use this function. I add global_step, so that can log the result
    # of valid to tensorboard

    world_size = util.get_world_size()
    rank = util.get_rank()

    # test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    test_triplets = test_data.statements
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    # build a dict from the triplets, the dict is used to compute the negative
    if cfg.task.n_ary_negative:
        gt_triplets = test_data.gt_statements
        hr2t_dict = {}
        tr2h_dict = {}
        for i in range(gt_triplets.shape[0]):
            statement_i = gt_triplets[i, :]
            h, t, r, quals = statement_i[0], statement_i[1], statement_i[2], statement_i[3:]
            hr2t_dict[(int(h), int(r), *quals.tolist())] = int(t)
            tr2h_dict[(int(t), int(r), *quals.tolist())] = int(h)
    else:
        hr2t_dict = None
        tr2h_dict = None

    model.eval()
    rankings = []
    num_negatives = []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        batch = batch.t()
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch, hr2t_dict=hr2t_dict, tr2h_dict=tr2h_dict)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch, hr2t_dict=hr2t_dict, tr2h_dict=tr2h_dict)
        pos_h_index, pos_t_index, pos_r_index = batch[0], batch[1], batch[2]
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            if global_step is not None and split is not None:
                writer.add_scalar(split + "/" + metric, score, global_step)
    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    cfg['experiment_name'] = args.name
    if not "n_ary_negative" in cfg.task:
        cfg.task.n_ary_negative = False
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        writer = SummaryWriter("./")
    is_inductive = cfg.dataset["class"].startswith("Ind") or "inductive" in cfg.dataset["class"].lower()
    dataset = util.build_nary_dataset(cfg)
    # some attribute of model is depending on dataset
    try:
        cfg.model.nbf_config.num_relation = dataset.num_relations
    except:
        cfg.model.starE_nbf_config.num_relation = dataset.num_relations
    cfg.model.hyper_relation_learner_config.num_relation = dataset.hyper_num_relations
    cfg.model.hyper_relation_learner_config.num_entity = dataset.hyper_num_nodes
    cfg.model.hyper_relation_learner_config.statement_len = dataset.maxlen
    if cfg.model.hyper_relation_learner_config.use_qual_embedding:
        cfg.model.hyper_relation_learner_config.num_qual_relation = dataset.hyper_num_qual_relations
    model = util.build_nary_model(cfg)

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    # TODO: check here
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        # filtered_data = Data(edge_index=dataset.data.edge_index, edge_type=dataset.data.edge_type)
        # filtered_data = filtered_data.to(device)
        # for RelLinkPredDataset, the data.edge_index and data.edge_type is from the train.txt
        #  (this may be needed because the train.txt contains all entities, while the test/valid.txt not)
        filtered_data = None

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data, test_data=test_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data, global_step=cfg.train.num_epoch + 1, split="valid")
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data, global_step=cfg.train.num_epoch + 1, split="test")

    if util.get_rank() == 0:
        writer.close()

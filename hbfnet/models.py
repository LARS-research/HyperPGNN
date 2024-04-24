from torch import nn
from nbfnet.models import NBFNet
from nbfnet.stare_nbfnet import StarENBFNet
from hbfnet.hyper_relation_learner import HyperRelationLearner, un_pad
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from hbfnet.utils import get_param

class HyperBFNet(nn.Module):
    def __init__(self, nbf_config, hyper_relation_learner_config, score_config):
        super(HyperBFNet, self).__init__()
        self.nbf_config = nbf_config
        self.hyper_relation_learner_config = hyper_relation_learner_config
        self.score_config = score_config

        # NBFNet
        if "p" not in nbf_config:
            self.starE_nbf = False
            self.nbf = NBFNet(**nbf_config)
        else:
            self.starE_nbf = True
            self.nbf = StarENBFNet(**nbf_config)

        # In-corp
        if hyper_relation_learner_config is not None:
            if "transformer_config" not in hyper_relation_learner_config:
                self.use_transformer = False
                self.incorp = HyperRelationLearner(**hyper_relation_learner_config)
            else:
                self.use_transformer = True
                # build a transformer based on transformer_config
                transformer_config = hyper_relation_learner_config["transformer_config"]
                transformer_layer = TransformerEncoderLayer(d_model=transformer_config["d_model"],
                                                            nhead=transformer_config["nhead"],
                                                            dim_feedforward=transformer_config["dim_feedforward"],
                                                            dropout=transformer_config["dropout"])
                self.transformer = TransformerEncoder(transformer_layer, num_layers=transformer_config["num_layers"])
                if transformer_config["use_positional_encoding"]:
                    self.position_embeddings = nn.Embedding(transformer_config["max_len"] - 1,
                                                            transformer_config["d_model"])
                else:
                    self.position_embeddings = None
                self.qual_rel_embed = get_param((hyper_relation_learner_config["num_qual_relation"],
                                            hyper_relation_learner_config["input_dim"]))
                # here the last embedding is for padding, initialize to be zero tensor
                self.ent_embed = get_param((hyper_relation_learner_config["num_entity"], hyper_relation_learner_config["input_dim"]))
                self.rel_embed = get_param((hyper_relation_learner_config["num_relation"]*2,
                                            hyper_relation_learner_config["input_dim"]))

        self.version = hyper_relation_learner_config.version

        # MLP scoring
        hidden_dims = nbf_config.hidden_dims
        input_dim = nbf_config.input_dim
        concat_hidden = score_config.concat_hidden
        num_mlp_layer = score_config.num_mlp_layer
        if self.version == "v0":
            feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim * 3
        else:
            feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, data, batch):
        # imitate nbfnet\models.py forward()
        # data is graph, batch is h_index, t_index, r_index
        # ****************** preprocessing from NBFNet******************
        # h_index, t_index, r_index = batch.unbind(-1)
        h_index, t_index, r_index = batch

        # for NBFNet, need split the r_index, now r_index contrains qualifiers
        shape = h_index.shape
        quals = r_index[:, shape[1]:]
        r_index = r_index[:, :shape[1]]

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            if not self.starE_nbf:
                data = self.nbf.remove_easy_edges(data, h_index, t_index, r_index)
            else:
                pass

        # TODO: where is the add_inverse? NBFNet has this, did NBFNet-PyG remove this?
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.nbf.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # ****************** Hyper relation learner ******************
        # this is the so-called Perp
        if not self.use_transformer:
            query_embedding, x, r = self.incorp(quals, r_index,
                                                hypergraph_edge_index=data.hyper_edge_index,
                                                hypergraph_edge_type=data.hyper_edge_type,
                                                hypergraph_quals=data.hyper_quals)
        else:
            quals_qr = torch.arange(start=0, end=quals.shape[1], step=2, device=quals.device)
            quals_qe = torch.arange(start=1, end=quals.shape[1], step=2, device=quals.device)
            qr_index = torch.index_select(quals, dim=1, index=quals_qr)
            qe_index = torch.index_select(quals, dim=1, index=quals_qe)

            qual_mask= qr_index == -1
            qr_index[qr_index==-1] = self.qual_rel_embed.shape[0]
            qe_index[qe_index==-1] = self.ent_embed.shape[0]
            qual_rel_embed = torch.cat((self.qual_rel_embed, 
                                        torch.zeros(1, self.qual_rel_embed.shape[1], 
                                                    requires_grad=False, device=self.qual_rel_embed.device)), dim=0)
            qr_embedding = qual_rel_embed[qr_index]  # batch * num_qual * dim
            ent_embed = torch.cat((self.ent_embed, 
                                   torch.zeros(1, self.ent_embed.shape[1], 
                                               requires_grad=False, device=self.ent_embed.device)), dim=0)
            qe_embedding = ent_embed[qe_index]  # batch * num_qual * dim
            
            r_embedding = self.rel_embed[r_index[:,0]].view(-1, 1, qr_embedding.shape[-1])
            quals_embedding = torch.cat((qr_embedding, qe_embedding), 2).view(-1, 2*qr_embedding.shape[1],
                                                                              qr_embedding.shape[2])
            stk_inp = torch.cat([r_embedding, quals_embedding], 1).transpose(1, 0) # [1 + num_qual_pairs, bs, 2*dim]
            # add positional encoding
            if self.position_embeddings is not None:
                positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
                pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
                stk_inp = stk_inp + pos_embeddings
            
            rel_mask = torch.zeros(stk_inp.shape[1], 1, device=self.device).bool()
            seq_mask = torch.cat((rel_mask, qual_mask, qual_mask), dim=1)

            transformer_output = self.transformer(stk_inp, src_key_padding_mask=seq_mask)
            # select the first token
            # select the mean
            query_embedding = torch.mean(transformer_output, dim=0)

            x = ent_embed
            r = self.rel_embed

        # ****************** NBFNet forward main ******************
        # message passing and updated node representations
        if self.starE_nbf:
            data.hyper_edge_index = data.hyper_edge_index.T  # it seems the edge_index used by nbf is of the shape [2, num_of_edges],
            # while the edge_index used by starE is of the shape [num_of_edges, 2]?
            if not self.use_transformer:
                output = self.nbf.bellmanford(data, h_index[:, 0], r_index[:, 0],
                                        query_embedding=query_embedding,
                                        initial_x = self.incorp.ent_embed,
                                        quals=data.hyper_quals, 
                                        rel_embed=self.incorp.rel_embed, qual_rel_embed=self.incorp.qual_rel_embed)
            else:
                output = self.nbf.bellmanford(data, h_index[:, 0], r_index[:, 0],
                                        query_embedding=query_embedding,
                                        initial_x = self.ent_embed,
                                        quals=data.hyper_quals, 
                                        rel_embed=self.rel_embed, qual_rel_embed=self.qual_rel_embed)
        else:
            output = self.nbf.bellmanford(data, h_index[:, 0], r_index[:, 0],
                                          query_embedding=query_embedding)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        # score = self.mlp(feature).squeeze(-1)

        if self.score_config.symmetric:
            # assert (t_index[:, [0]] == t_index).all()
            output = self.nbf.bellmanford(data, t_index[:, 0], r_index[:, 0])
            inv_feature = output["node_feature"].transpose(0, 1)  # TODO: perhaps the output of NBFNet-PyG not use transpose
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2
        if self.version == "v0":
            feature = torch.cat((x[h_index], r[r_index], feature), dim=-1)
            score = self.mlp(feature).squeeze(-1)
        else:
            score = self.mlp(feature).squeeze(-1)  # (B * number_of_fact_per_batch)
            # score = score.view(shape[0], shape[1])  # (B, number_of_fact_per_batch)
        return score.view(shape)
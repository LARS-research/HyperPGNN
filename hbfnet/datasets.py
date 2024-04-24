import os

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import csv
from tqdm import tqdm
from hbfnet.hyper_relation_learner import un_pad


class BaseHyper2TriDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @staticmethod
    def pad_statements(statements, maxlen):
        result = [statement + [-1] * (maxlen - len(statement)) if len(statement) < maxlen
                  else statement[:maxlen] for statement in statements]
        return result

    # THIS IS COPIED FROM TORCHDRUG
    # def load_triplet(self, triplets, entity_vocab=None, relation_vocab=None, inv_entity_vocab=None,
    #                  inv_relation_vocab=None):
    #     """
    #     Load the dataset from triplets.
    #     The mapping between indexes and tokens is specified through either vocabularies or inverse vocabularies.
    #
    #     Parameters:
    #         triplets (array_like): triplets of shape :math:`(n, 3)`
    #         entity_vocab (dict of str, optional): maps entity indexes to tokens
    #         relation_vocab (dict of str, optional): maps relation indexes to tokens
    #         inv_entity_vocab (dict of str, optional): maps tokens to entity indexes
    #         inv_relation_vocab (dict of str, optional): maps tokens to relation indexes
    #     """
    #     entity_vocab, inv_entity_vocab = self._standarize_vocab(entity_vocab, inv_entity_vocab)
    #     relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, inv_relation_vocab)
    #
    #     num_node = len(entity_vocab) if entity_vocab else None
    #     num_relation = len(relation_vocab) if relation_vocab else None
    #     self.graph = data.Graph(triplets, num_node=num_node, num_relation=num_relation)
    #     self.entity_vocab = entity_vocab
    #     self.relation_vocab = relation_vocab
    #     self.inv_entity_vocab = inv_entity_vocab
    #     self.inv_relation_vocab = inv_relation_vocab
    #
    # def load_qualifiers(self, statements, relation_vocab=None, inv_relation_vocab=None):
    #     """
    #     Load the dataset from statements.
    #     The mapping between indexes and tokens is specified through either vocabularies or inverse vocabularies.
    #
    #     Parameters:
    #         statements (array_like): statements of shape :math:`(n, maxlen)`
    #         relation_vocab (dict of str, optional): maps relation indexes to tokens
    #         inv_relation_vocab (dict of str, optional): maps tokens to relation indexes
    #     """
    #     relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, inv_relation_vocab)
    #     self.statements = statements
    #     self.qual_relation_vocab = relation_vocab
    #     self.inv_qual_relation_vocab = inv_relation_vocab

    # These are for starE
    @staticmethod
    def get_repr(available_triplets, available_statements, num_relation, add_reverse=False):
        h_index, t_index, r_index = available_triplets.t()
        edge_index = torch.stack([torch.cat([h_index, t_index]),
                                  torch.cat([t_index, h_index])], dim=1).squeeze()
        edge_type = torch.cat([r_index, r_index + num_relation]).squeeze()
        quals = torch.tensor(available_statements)[:, 3:]
        quals = un_pad(quals)

        # liushuzhi add, refer to get_alternative_graph_repr in StarE
        # quals = torch.concat([quals, quals], dim=1)
        # TODO: uncomment this!
        if add_reverse:
            quals = torch.cat([quals, quals], dim=1)
        return edge_index, edge_type, quals

    @staticmethod
    def read_main_from_txt(tsv_file, inv_entity_vocab, inv_relation_vocab, delimiter=","):
        triplets = []
        maxlen = 0
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter=delimiter)

            num_sample = 0
            for tokens in reader:
                h_token, r_token, t_token = tokens[:3]
                if h_token not in inv_entity_vocab:
                    inv_entity_vocab[h_token] = len(inv_entity_vocab)
                h = inv_entity_vocab[h_token]
                if r_token not in inv_relation_vocab:
                    inv_relation_vocab[r_token] = len(inv_relation_vocab)
                r = inv_relation_vocab[r_token]
                if t_token not in inv_entity_vocab:
                    inv_entity_vocab[t_token] = len(inv_entity_vocab)
                t = inv_entity_vocab[t_token]
                triplets.append((h, t, r))
                num_sample += 1
                if len(tokens) > maxlen:
                    maxlen = len(tokens)
        return triplets, inv_entity_vocab, inv_relation_vocab, maxlen  # , num_sample,

    @staticmethod
    def read_qualifiers_from_txt(tsv_file, inv_entity_vocab, inv_relation_vocab, inv_qual_relation_vocab, delimiter=","):
        # will update the input vocab

        statements = []
        mask = []
        triplets = []
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter=delimiter)
            for tokens in reader:
                if delimiter == "\t":
                    tokens[:] = [x for x in tokens if x]  # remove '' when reading path_graph.
                h_token, r_token, t_token = tokens[:3]
                h = inv_entity_vocab[h_token]
                r = inv_relation_vocab[r_token]
                t = inv_entity_vocab[t_token]
                tmp = [h, t, r]
                if len(tokens) > 3:
                    for i in range(3, len(tokens), 2):
                        qr_token, qe_token = tokens[i:i + 2]
                        if qe_token not in inv_entity_vocab:
                            # mask seems to be inidcate the qualifiers contain entities not seen in "main triplets"
                            inv_entity_vocab[qe_token] = len(inv_entity_vocab)
                            mask += [len(inv_entity_vocab) - 1]
                        qe = inv_entity_vocab[qe_token]
                        if qr_token not in inv_qual_relation_vocab:
                            inv_qual_relation_vocab[qr_token] = len(inv_qual_relation_vocab)
                        qr = inv_qual_relation_vocab[qr_token]
                        tmp = tmp + [qr, qe]

                        if (r_token, qr_token) not in inv_relation_vocab:
                            inv_relation_vocab[(r_token, qr_token)] = len(inv_relation_vocab)
                        qr = inv_relation_vocab[(r_token, qr_token)]
                        # Hyper2Tri
                        triplets.append((h, qe, qr))
                        triplets.append((t, qe, qr))

                        for j in range(3, i - 1, 2):
                            qr_t, qe_t = tokens[j:j + 2]
                            qe_t = inv_entity_vocab[qe_t]
                            if (qr_token, qr_t) not in inv_relation_vocab:
                                inv_relation_vocab[(qr_token, qr_t)] = len(inv_relation_vocab)
                            qr_t = inv_relation_vocab[(qr_token, qr_t)]
                            triplets.append((qe, qe_t, qr_t))

                statements.append(tmp)
        return statements, inv_qual_relation_vocab, mask, triplets, inv_entity_vocab, inv_relation_vocab

    def gather_all_answers_to_query(self, data):
        hyper_edge_index = data.hyper_edge_index
        hyper_edge_type = data.hyper_edge_type
        hyper_quals = data.hyper_quals


    @staticmethod
    def add_reverse_for_nbf(edge_index, edge_type, num_relations):
        # NBF-PyG did not call graph.undirected(add_inverse=True) as the torchdrug version did.
        # so need to add reverse edges in dataset.
        row, col = edge_index
        edge_type = edge_type
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type +num_relations])
        return edge_index, edge_type




class NAryLinkPrediction(BaseHyper2TriDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.num_relations = int(self._data.edge_type.max().item()) + 1  # NBFNet will use this
        # self.maxlen = int(self._data.maxlen.max().item())
        # self.hyper_num_relations = int(self._data.hyper_num_relations[0].item())  # StarE will use this, is from
        # # train_data
        # self.hyper_num_nodes = int(self._data.hyper_num_nodes[0].item())  # StarE will use this, is from train_data
        # self.hyper_num_qual_relations = int(self._data.hyper_num_qual_relations[0].item())

        self.num_relations = int(self.data.edge_type.max().item()) + 1  # NBFNet will use this
        self.maxlen = int(self.data.maxlen.max().item())
        self.hyper_num_relations = int(self.data.hyper_num_relations[0].item())  # StarE will use this, is from
        # train_data
        self.hyper_num_nodes = int(self.data.hyper_num_nodes[0].item())  # StarE will use this, is from train_data
        self.hyper_num_qual_relations = int(self.data.hyper_num_qual_relations[0].item())

    @property
    def raw_file_names(self):
        return [
            "train.txt", "valid.txt", "test.txt"
        ]

    def process(self):
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        # load all the main triplets
        triplets_main_train, inv_entity_vocab, inv_relation_vocab, maxlen_train = \
            self.read_main_from_txt(self.raw_paths[0], inv_entity_vocab, inv_relation_vocab)
        triplets_main_valid, inv_entity_vocab, inv_relation_vocab, maxlen_valid = \
            self.read_main_from_txt(self.raw_paths[1], inv_entity_vocab, inv_relation_vocab)
        triplets_main_test, inv_entity_vocab, inv_relation_vocab, maxlen_test = \
            self.read_main_from_txt(self.raw_paths[2], inv_entity_vocab, inv_relation_vocab)

        # load the qualifiers
        self.num_main_rel = len(inv_relation_vocab)
        inv_qual_relation_vocab = {}  # jianwen use separate embedding for relations in qualifiers, still share between
        # train and test
        mask = []  # TODO: what is the mask here?
        statements_train, inv_qual_relation_vocab, mask_train, triplets_train_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[0], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)
        statements_valid, inv_qual_relation_vocab, mask_valid, triplets_valid_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[1], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)
        statements_test, inv_qual_relation_vocab, mask_test, triplets_test_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[2], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)

        # self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
        # # this is inherited from KnowledgeGraphDataset， build graph containing all triplets.
        maxlen = max(maxlen_train, maxlen_valid, maxlen_test)
        statements_train = self.pad_statements(statements_train, maxlen)
        statements_valid = self.pad_statements(statements_valid, maxlen)
        statements_test = self.pad_statements(statements_test, maxlen)

        mask = torch.tensor(mask_train + mask_valid + mask_test)  # This is for test. Mask nodes that only apper in quals from when
        # testing. The mask is used for the hyper2Tri graph, when the hyper2tri graph is shared by train/valid/test,
        # this should also be shared.

        # self.load_qualifiers(statements, inv_relation_vocab=inv_qual_relation_vocab)
        # self.available_statements = self.pad_statements(available_statements, maxlen)
        # self.num_samples = num_samples
        # self.mask = mask
        # self.num_qual_relation = len(inv_qual_relation_vocab)
        # self.get_repr()
        # triplets = torch.tensor(triplets)

        # edge_index = triplets[:, :2].t()
        # edge_type = triplets[:, 2]
        # num_relations = int(edge_type.max()) + 1

        # For NBFNet, sort of "fact_graph"
        triplets_train = torch.tensor(triplets_main_train + triplets_train_quals)
        # triplets_train = torch.tensor(triplets_main_train + triplets_train_quals
        #                               + triplets_valid_quals + triplets_test_quals)
        train_edge_index = triplets_train[:, :2].t()
        train_edge_type = triplets_train[:, 2]
        train_edge_index, train_edge_type = self.add_reverse_for_nbf(train_edge_index, train_edge_type, len(inv_relation_vocab))
        # num_nodes_of_train = len(set(train_edge_index.flatten().to_list()))
        num_nodes_of_train = train_edge_index.max() + 1
        # It maybe better to use the train_edge_index.max()+1, because the train_edge_index may not be continuous.
        # num_edges_of_train = len(set(train_edge_type.flatten().to_list()))
        # It maybe better to use the edge_type.max()+1, because the edge_type may not be continuous.
        # TODO: the edge_index and edge_type may not be continuous, perhaps reindex them.

        # For StarE, corresponds to Jianwen's get_repr
        available_triplets = torch.tensor(triplets_main_train)
        hyper_edge_index, hyper_edge_type, quals = \
            self.get_repr(available_triplets, statements_train, len(inv_relation_vocab))  # this follows Jianwen
        # TODO: here the indexing inside statements the is those containing hyper2tri results.
        # TODO: here the number of relation is those containing hyper2tri relation.
        hyper_num_nodes = len(inv_entity_vocab)
        hyper_num_relations = len(inv_relation_vocab)
        hyper_num_qual_relation = len(inv_qual_relation_vocab)
        # TODO: if not use separate embedding for r/qualifiers, this will not be used.

        # for compute mask, this corresponds to jianwen's graph, no reveser edge is added
        whole_triplets = torch.tensor(triplets_main_train + triplets_main_valid + triplets_main_test +
                                      triplets_train_quals + triplets_valid_quals + triplets_test_quals)
        whole_edge_index = whole_triplets[:, :2].t()
        whole_edge_type = whole_triplets[:, 2]
        whole_num_nodes = whole_edge_index.max()+1

        # Note, edge_index, edge_type, num_nodes is used by NBFNet, it is the normal graph after Hyper2Tri, do not have
        # hyper graph.
        # Statements stored the facts. Note the edge in statements may appear in the graph structure, so we need to
        # remove them when used by NBFNet.
        # hyper_edge_index, hyper_edge_type, hyper_quals are used by starE(Prep), the hyper_num_nodes,
        # hyper_num_relations, hyper_num_qual_relations are used by StarE
        # hyper_edge_index, hyper_edge_type, hyper_quals are also used to produce answers to all [h,s,?, qualifiers] and
        # [?,s,t, qualifiers] queries.

        gt_statements_train = torch.tensor(statements_train)
        gt_statements_valid = torch.tensor(statements_train + statements_valid)
        gt_statements_test = torch.tensor(statements_train + statements_test)  # TODO: add valid?

        train_data = Data(edge_index=train_edge_index, edge_type=train_edge_type, num_nodes=num_nodes_of_train,
                          statements=torch.tensor(statements_train), maxlen=maxlen, mask=mask,
                          hyper_edge_index=hyper_edge_index, hyper_edge_type=hyper_edge_type, hyper_quals=quals,
                          hyper_num_nodes=hyper_num_nodes, hyper_num_relations=hyper_num_relations,
                          hyper_num_qual_relations=hyper_num_qual_relation,
                          whole_edge_index=whole_edge_index, whole_edge_type=whole_edge_type, 
                          whole_num_nodes=whole_num_nodes,
                          gt_statements=gt_statements_train)
        valid_data = Data(edge_index=train_edge_index, edge_type=train_edge_type, num_nodes=num_nodes_of_train,
                          statements=torch.tensor(statements_valid), maxlen=maxlen, mask=mask,
                          hyper_edge_index=hyper_edge_index, hyper_edge_type=hyper_edge_type, hyper_quals=quals,
                          hyper_num_nodes=hyper_num_nodes, hyper_num_relations=hyper_num_relations,
                          hyper_num_qual_relations=hyper_num_qual_relation,
                          whole_edge_index=whole_edge_index, whole_edge_type=whole_edge_type, 
                          whole_num_nodes=whole_num_nodes,
                          gt_statements=gt_statements_valid)
        test_data = Data(edge_index=train_edge_index, edge_type=train_edge_type, num_nodes=num_nodes_of_train,
                         statements=torch.tensor(statements_test), maxlen=maxlen, mask=mask,
                         hyper_edge_index=hyper_edge_index, hyper_edge_type=hyper_edge_type, hyper_quals=quals,
                         hyper_num_nodes=hyper_num_nodes, hyper_num_relations=hyper_num_relations,
                         hyper_num_qual_relations=hyper_num_qual_relation,
                          whole_edge_index=whole_edge_index, whole_edge_type=whole_edge_type, 
                          whole_num_nodes=whole_num_nodes,
                          gt_statements=gt_statements_test)


        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name


class NaryLinkPredictionInductive(BaseHyper2TriDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_relations = int(self._data.edge_type.max().item()) + 1  # NBFNet will use this
        import pdb
        pdb.set_trace()
        self.maxlen = int(self._data.maxlen.max().item())
        self.hyper_num_relations = int(self._data.hyper_num_relations[0].item())  # StarE will use this, is from
        # train_data
        self.hyper_num_nodes = int(self._data.hyper_num_nodes[0].item())  # StarE will use this, is from train_data
        self.hyper_num_qual_relations = int(self._data.hyper_num_qual_relations[0].item())

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return [
            "train.txt", "valid.txt", "test.txt", "aux.txt"
        ]

    def process(self):
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        # load all the main triplets
        triplets_main_train, inv_entity_vocab, inv_relation_vocab, maxlen_train = \
            self.read_main_from_txt(self.raw_paths[0], inv_entity_vocab, inv_relation_vocab)
        triplets_main_valid, inv_entity_vocab, inv_relation_vocab, maxlen_valid = \
            self.read_main_from_txt(self.raw_paths[1], inv_entity_vocab, inv_relation_vocab)
        triplets_main_test, inv_entity_vocab, inv_relation_vocab, maxlen_test = \
            self.read_main_from_txt(self.raw_paths[2], inv_entity_vocab, inv_relation_vocab)
        triplets_main_aux, inv_entity_vocab, inv_relation_vocab, maxlen_aux = \
            self.read_main_from_txt(self.raw_paths[3], inv_entity_vocab, inv_relation_vocab)

        # load the qualifiers
        self.num_main_rel = len(inv_relation_vocab)
        inv_qual_relation_vocab = {}  # jianwen use separate embedding for relations in qualifiers, still share between
        # train and test
        mask = []  # TODO: what is the mask here?
        statements_train, inv_qual_relation_vocab, mask_train, triplets_train_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[0], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)
        statements_valid, inv_qual_relation_vocab, mask_valid, triplets_valid_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[1], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)
        statements_test, inv_qual_relation_vocab, mask_test, triplets_test_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[2], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)
        statements_aux, inv_qual_relation_vocab, mask_aux, triplets_aux_quals, inv_entity_vocab, \
            inv_relation_vocab = self.read_qualifiers_from_txt(self.raw_paths[3], inv_entity_vocab,
                                                               inv_relation_vocab, inv_qual_relation_vocab)

        maxlen = max(maxlen_train, maxlen_valid, maxlen_test, maxlen_aux)
        statements_train = self.pad_statements(statements_train, maxlen)
        statements_valid = self.pad_statements(statements_valid, maxlen)
        statements_test = self.pad_statements(statements_test, maxlen)
        statements_aux = self.pad_statements(statements_aux, maxlen)

        # For NBFNet
        # triplets_train = torch.tensor(triplets_main_train + triplets_train_quals)
        triplets_train = torch.tensor(triplets_main_train + triplets_train_quals)
        train_edge_index = triplets_train[:, :2].t()
        train_edge_type = triplets_train[:, 2]
        num_nodes_of_train = train_edge_index.max() + 1
        train_edge_index, train_edge_type = self.add_reverse_for_nbf(train_edge_index, train_edge_type,
                                                                     len(inv_relation_vocab))

        triplets_test = torch.tensor(triplets_main_train + triplets_train_quals +  # TODO: do I need train?
                                     triplets_main_test + triplets_test_quals +  # TODO: Note I abort valid here.
                                     triplets_main_aux + triplets_aux_quals)

        test_edge_index = triplets_test[:, :2].t()
        test_edge_type = triplets_test[:, 2]
        num_nodes_of_test = test_edge_index.max() + 1
        test_edge_index, test_edge_type = self.add_reverse_for_nbf(test_edge_index, test_edge_type,
                                                                   len(inv_relation_vocab))

        # For StarE
        available_triplets = torch.tensor(triplets_main_train)
        # hyper_edge_index, hyper_edge_type, quals = \
        #     self.get_repr(available_triplets, statements_train, len(inv_relation_vocab))  # this follows Jianwen
        hyper_num_nodes = len(inv_entity_vocab)
        hyper_num_relations = len(inv_relation_vocab)
        hyper_num_qual_relation = len(inv_qual_relation_vocab)

        # note: for inductive setting, the num_relations is shared for train/test.
        train_data = Data(edge_index=train_edge_index, edge_type=train_edge_type, num_nodes=num_nodes_of_train,
                          statements=torch.tensor(statements_train), maxlen=maxlen,
                          hyper_edge_index=[], hyper_edge_type=[], hyper_quals=[],
                          hyper_num_nodes=hyper_num_nodes, hyper_num_relations=hyper_num_relations,
                          hyper_num_qual_relations=hyper_num_qual_relation)
        valid_data = Data(edge_index=train_edge_index, edge_type=train_edge_type, num_nodes=num_nodes_of_train,
                          statements=torch.tensor(statements_valid), maxlen=maxlen,
                          hyper_edge_index=[], hyper_edge_type=[], hyper_quals=[],
                          hyper_num_nodes=hyper_num_nodes, hyper_num_relations=hyper_num_relations,
                          hyper_num_qual_relations=hyper_num_qual_relation)
        test_data = Data(edge_index=test_edge_index, edge_type=test_edge_type, num_nodes=num_nodes_of_test,
                         statements=torch.tensor(statements_test), maxlen=maxlen,
                         hyper_edge_index=[], hyper_edge_type=[], hyper_quals=[],  # just a placeholder, so that the
                         # model of inductive can be unified with the transductive one.
                         hyper_num_nodes=hyper_num_nodes, hyper_num_relations=hyper_num_relations,
                         hyper_num_qual_relations=hyper_num_qual_relation  # although here is hyper_num_nodes, only
                         # them in qualifiers is used, so still inductive. The nodes/relations in qualifiers and main_
                         # relations is not inductive
                         )

        import pdb
        pdb.set_trace()

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name

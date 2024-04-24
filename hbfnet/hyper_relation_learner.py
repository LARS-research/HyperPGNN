from torch import nn
from hbfnet.layers import EdgeProcessing
from hbfnet.utils import get_param, ccorr, rotate
from torch_scatter import scatter_add, scatter_mean
import torch


def un_pad(quals):
    ent = []
    rel = []
    idx = []
    for i, qual in enumerate(quals):
        for j in range(0, len(qual) - 1, 2):
            if qual[j] != -1:
                rel.append(qual[j])
                ent.append(qual[j+1])
                idx.append(i)
            else: break
    rel, ent, idx = torch.tensor(rel), torch.tensor(ent), torch.tensor(idx)
    result = torch.stack((rel, ent, idx), dim=0)
    return result


class HyperRelationLearner(nn.Module):

    def __init__(self, input_dim, output_dim, num_relation,
                 num_entity,  # TODO, where to set these? They are constructed by the graph.
                 # TODO: the num_entity, num_relation and num_qual_relation should be set according to the hypergraph
                 # edge_index, edge_type, quals,
                 statement_len, opn="rotate", qual_aggregate='sum', qual_n='sum', alpha=0.8, gcn_drop=0.1, bias=False,
                 version=None, use_qual_embedding=False, num_qual_relation=0):
        super(HyperRelationLearner, self).__init__()


        self.p = {
            'STATEMENT_LEN': statement_len,
            'OPN': opn,
            'QUAL_AGGREGATE': qual_aggregate,
            'QUAL_N': qual_n,
            'TRIPLE_QUAL_WEIGHT': alpha,
            'GCN_DROP': gcn_drop,
            'BIAS': bias,
        }
        self.statement_len = statement_len
        self.edgeprocess = EdgeProcessing(input_dim, output_dim, num_relation, torch.tanh, self.p)
        self.version = version

        num_relation = int(num_relation)
        double_relation = num_relation * 2  # for inverse relation. So that the input num_relation should be the num
        # w/o inverse relations.
        self.rel_embed = get_param((double_relation, input_dim))  # TODO: remove this+1, add +1 to qual_rel_embed
        assert use_qual_embedding; "use_qual_embedding must be True, currently my dataset use separate index for relations in quals"
        if use_qual_embedding:
            self.qual_rel_embed = get_param((num_qual_relation + 1, input_dim))
        else:
            self.qual_rel_embed = None
        self.ent_embed = get_param((num_entity, input_dim))

        if self.p['STATEMENT_LEN'] != 3:
            if self.p['QUAL_AGGREGATE'] == 'sum' or self.p['QUAL_AGGREGATE'] == 'mul':
                self.w_q = get_param((input_dim, input_dim))  # new for quals setup
            elif self.p['QUAL_AGGREGATE'] == 'concat':
                self.w_q = get_param((2 * input_dim, input_dim))  # need 2x size due to the concat operation

    def forward(self, quals, r_index, hypergraph_edge_index, hypergraph_edge_type, hypergraph_quals, new_relation_embedding=None):
        # quals and r_index is the input qualifier and relation.
        #   not the quals here is different from the quals of the whold graph, which is used by self.edgeprocess

        # device = r_index.device
        # This is so-called Perp, message passing on the whole graph, not related to the input query
        if self.version == "wo_incorp_prep" or self.version == "wo_prep":
            x = self.ent_embed
            r = self.rel_embed
        else:
            if self.statement_len == 3:
                x, r = self.edgeprocess(x=self.ent_embed, edge_index=hypergraph_edge_index,
                                        edge_type=hypergraph_edge_type, rel_embed=self.rel_embed,
                                        quals=None, qual_rel_embed=self.qual_rel_embed)
            else:
                x, r = self.edgeprocess(x=self.ent_embed, edge_index=hypergraph_edge_index,
                                        edge_type=hypergraph_edge_type, rel_embed=self.rel_embed,
                                        quals=hypergraph_quals, qual_rel_embed=self.qual_rel_embed)

        # gettig query from x[self.num_entity:]
        # THis is the so-called InCorp module
        # TODO: this is sort of update_rel_emb_with_qualifier, starE use this as part of message function.
        if self.statement_len > 3:
            quals = un_pad(quals).long().to(r_index.device)
            if self.qual_rel_embed is not None:
                qualifier_emb_rel = self.qual_rel_embed[quals[0]]
                # TODO: do not use self.qual_rel_embed
            else:
                qualifier_emb_rel = self.rel_embed[quals[0]]
            if self.version == "v0" or self.version == "wo_prep":
                qualifier_emb_ent = self.ent_embed[quals[1]]  #################################
                qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                                    qualifier_rel=qualifier_emb_rel)
                if new_relation_embedding is not None:
                    rel_part_emb = new_relation_embedding
                else:
                    rel_part_emb = self.rel_embed[r_index[:, 0]]  #################################
                query_embedding = self.qualifier_aggregate(qualifier_emb, rel_part_emb,
                                                           alpha=self.p['TRIPLE_QUAL_WEIGHT'],
                                                           qual_index=quals[2])
            elif self.version == "v1":
                qualifier_emb_ent = x[quals[1]]  #################################
                qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                                    qualifier_rel=qualifier_emb_rel)
                if new_relation_embedding is not None:
                    rel_part_emb = new_relation_embedding.expand(r_index.shape[0], -1)
                else:
                    rel_part_emb = r[r_index[:, 0]]
                query_embedding = self.qualifier_aggregate(qualifier_emb, rel_part_emb,
                                                           alpha=self.p['TRIPLE_QUAL_WEIGHT'],
                                                           qual_index=quals[2])
            elif self.version == "wo_incorp_prep":
                query_embedding = "wo_incorp_prep"
            else:
                raise ValueError("version not supported")

            # rel_part_emb = self.new_relation_random_embedding.expand(h_index.shape[0], -1)

        else:
            if new_relation_embedding is not None:
                return new_relation_embedding.expand(r_index.shape[0], -1), x, r
            if self.version == "v0" or self.version == "wo_prep" or self.version == "wo_incorp_prep":
                query_embedding = self.rel_embed[r_index[:, 0]]
            elif self.version == "v1":
                query_embedding = r[r_index[:, 0]]
            else:
                raise ValueError("version not supported")

        return query_embedding, x, r


    # starE codes
    def qual_transform(self, qualifier_ent, qualifier_rel):
        """

        :return:
        """
        if self.p['OPN'] == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.p['OPN'] == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.p['OPN'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.p['OPN'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return trans_embed

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):
        """
            In qualifier_aggregate method following steps are performed

            qualifier_emb looks like -
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            rel_part_emb       :   [qq,ww,ee,rr,tt, .....]                      (here qq, ww, ee .. are of 200 dim)

            Note that rel_part_emb for jf17k would be around 61k*200

            Step1 : Pass the qualifier_emb to self.coalesce_quals and multiply the returned output with a weight.
            qualifier_emb   : [aa,bb,cc,dd,ee, ...... ]                 (here aa, bb, cc are of 200 dim each)
            Note that now qualifier_emb has the same shape as rel_part_emb around 61k*200

            Step2 : Combine the updated qualifier_emb (see Step1) with rel_part_emb based on defined aggregation strategy.



            Aggregates the qualifier matrix (3, edge_index, emb_dim)
        :param qualifier_emb:
        :param rel_part_emb:
        :param type:
        :param alpha
        :return:

        self.coalesce_quals    returns   :  [q+a+b+d,w+c+e+g,e'+f,......]        (here each element in the list is of 200 dim)

        """

        if self.p['QUAL_AGGREGATE'] == 'sum':
            qualifier_emb = torch.einsum('ij,jk -> ik',
                                         self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                                         self.w_q)
            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb  # [N_EDGES / 2 x EMB_DIM]
        elif self.p['QUAL_AGGREGATE'] == 'concat':
            qualifier_emb = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])
            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            return torch.mm(agg_rel, self.w_q)  # [N_EDGES / 2 x EMB_DIM]

        elif self.p['QUAL_AGGREGATE'] == 'mul':
            qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1),
                                     self.w_q)
            return rel_part_emb * qualifier_emb
        else:
            raise NotImplementedError

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """

        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            edge_type          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :param fill: fill value for the output matrix - should be 0 for sum/concat and 1 for mul qual aggregation strat
        :return: [1, N_EDGES]
        """

        if self.p['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
        elif self.p['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output
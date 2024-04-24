import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torchdrug.layers import functional
from torchdrug.utils import sparse_coo_tensor
from hbfnet.utils import get_param, ccorr, rotate
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min


class StarEGeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True, p=None):
        super(StarEGeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)

        # ********************** from starE **********************
        self.p = p

        self.w_loop = get_param((input_dim, output_dim))  # (100,200)
        self.w_in = get_param((input_dim, output_dim))  # (100,200)
        self.w_out = get_param((input_dim, output_dim))  # (100,200)
        self.w_rel = get_param((input_dim, output_dim))  # (100,200)

        if self.p['STATEMENT_LEN'] != 3:
            if self.p['QUAL_AGGREGATE'] == 'sum' or self.p['QUAL_AGGREGATE'] == 'mul':
                # self.w_q = nn.Embedding(in_channels, in_channels)  # new for quals setup
                self.w_q = get_param((input_dim, input_dim))  # new for quals setup
            elif self.p['QUAL_AGGREGATE'] == 'concat':
                # self.w_q = nn.Embedding(2 * in_channels, in_channels)  # need 2x size due to the concat operation
                self.w_q = get_param((2 * input_dim, input_dim))  # need 2x size due to the concat operation
        

        self.loop_rel = get_param((1, input_dim))  # (1,100)
        self.loop_ent = get_param((1, input_dim))  # new (seems no used)

        self.drop = torch.nn.Dropout(self.p['GCN_DROP'])
        self.bn = torch.nn.BatchNorm1d(output_dim)

        self.act = torch.tanh  # seem not set by the config
        # ********************** end of from starE **********************


    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None, 
                # add from stare
                x=None, quals=None, rel_embed=None, qual_rel_embed=None):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = self.relation.weight.expand(batch_size, -1, -1)
        # if edge_weight is None:
        #     edge_weight = torch.ones(len(edge_type), device=input.device)


        # ********************** from starE **********************
        device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)  # in starE, the loop belongs to new type of relation

        num_edges = len(edge_type) // 2
        # TODO: this is buggy. the input seems to be different from x?
        num_ent = input.shape[1]
        if edge_index.shape[0] > 2: edge_index = edge_index.t()

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, :num_edges]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        if self.p['STATEMENT_LEN'] != 3:
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1], quals[1]
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0], quals[0]
            self.quals_index_in, self.quals_index_out = quals[2], quals[2]

        '''
            Adding self loop by creating a COO matrix. Thus \
             loop index [1,2,3,4,5]
                        [1,2,3,4,5]
             loop type [10,10,10,10,10] --> assuming there are 9 relations


        '''
        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_edge_weight = torch.ones(len(self.in_type), device=input.device)
        loop_edge_weight = torch.ones(len(self.loop_type), device=input.device)
        out_edge_weight = torch.ones(len(self.out_type), device=input.device)

        if self.p['STATEMENT_LEN'] != 3 and len(quals[1]) > 0:

            in_res = self.propagate(edge_index=self.in_index,
                                    input=input,  # starE name it as x
                                    edge_type=self.in_type,
                                    rel_embed=rel_embed,
                                    qual_rel_embed=qual_rel_embed,
                                    edge_norm=self.in_norm,
                                    mode='in',
                                    ent_embed=x,
                                    qualifier_ent=self.in_index_qual_ent,
                                    qualifier_rel=self.in_index_qual_rel,
                                    qual_index=self.quals_index_in,
                                    source_index=self.in_index[0],
                                    # from nbfnet
                                    relation=relation, boundary=boundary, 
                                    size=size, edge_weight=in_edge_weight)

            loop_res = self.propagate(edge_index=self.loop_index,
                                      input=input,
                                      edge_type=self.loop_type,
                                      rel_embed=rel_embed,
                                      qual_rel_embed=qual_rel_embed,
                                      edge_norm=None,
                                      mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None,
                                      relation=relation, boundary=boundary, 
                                      size=size, edge_weight=loop_edge_weight)

            out_res = self.propagate(edge_index=self.out_index,
                                     input=input,
                                     edge_type=self.out_type,
                                     rel_embed=rel_embed,
                                     qual_rel_embed=qual_rel_embed,
                                     edge_norm=self.out_norm,
                                     mode='out',
                                     ent_embed=x,
                                     qualifier_ent=self.out_index_qual_ent,
                                     qualifier_rel=self.out_index_qual_rel,
                                     qual_index=self.quals_index_out,
                                     source_index=self.out_index[0],
                                     relation=relation, boundary=boundary, 
                                     size=size, edge_weight=out_edge_weight)

        else:  # for triplets
            in_res = self.propagate(edge_index=self.in_index,
                                    input=input,
                                    edge_type=self.in_type,
                                    rel_embed=rel_embed,
                                    qual_rel_embed=qual_rel_embed,
                                    edge_norm=self.in_norm,
                                    mode='in',
                                    ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                    qual_index=None, source_index=None,
                                    relation=relation, boundary=boundary, 
                                    size=size, edge_weight=in_edge_weight)

            loop_res = self.propagate(edge_index=self.loop_index,
                                      input=input,
                                      edge_type=self.loop_type,
                                      rel_embed=rel_embed,
                                      qual_rel_embed=qual_rel_embed,
                                      edge_norm=None,
                                      mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None,
                                      relation=relation, boundary=boundary, 
                                      size=size, edge_weight=loop_edge_weight)

            out_res = self.propagate(edge_index=self.out_index,
                                     input=input,
                                     edge_type=self.out_type,
                                     rel_embed=rel_embed,
                                     qual_rel_embed=qual_rel_embed,
                                     edge_norm=self.out_norm,
                                     mode='out',
                                     ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                     qual_index=None, source_index=None,
                                     relation=relation, boundary=boundary, 
                                     size=size, edge_weight=out_edge_weight)

        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p['BIAS']:
            out = out + self.bias
        # batch * num_ent * dim
        out = out.transpose(1, 2)   # batch * dim * num_ent
        out = self.bn(out)
        out = out.transpose(1, 2)   # batch * num_ent * dim

        # Ignoring the self loop inserted, return.
        # return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]
    
        return self.act(out)  # the rel_embed is not returned
        # ********************** end of from starE **********************

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        # output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
        #                         edge_type=edge_type, size=size, edge_weight=edge_weight)
        # return output

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(StarEGeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)
        else:
            return super(StarEGeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                     size, kwargs)
        # size = self._check_input(edge_index, size)
        # coll_dict = self._collect(self._fused_user_args, edge_index,
        #                           size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type,
                # starE added params
                rel_embed, qual_rel_embed, edge_norm, mode, ent_embed,
                qualifier_ent, qualifier_rel, qual_index, source_index):
        # TODO: rel_embed seem to eqaul to relation
        # TODO: self.p['OPN'] seems to be equal to self.message_func
        weight = getattr(self, 'w_{}'.format(mode))

        if self.p['STATEMENT_LEN'] != 3 and qualifier_rel is not None:
            if mode != 'loop':
                rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                            qualifier_rel, edge_type, qual_index,
                                                            qual_rel_embed=qual_rel_embed)
            else:
                rel_emb = torch.index_select(rel_embed, 0, edge_type)
        else:
            rel_emb = torch.index_select(rel_embed, 0, edge_type)
        relation_j = rel_emb
        # relation_j = relation.index_select(self.node_dim, edge_type)

        # if self.message_func == "transe":
        #     message = input_j + relation_j
        # elif self.message_func == "distmult":
        #     message = input_j * relation_j
        # elif self.message_func == "rotate":
        #     x_j_re, x_j_im = input_j.chunk(2, dim=-1)
        #     r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
        #     message_re = x_j_re * r_j_re - x_j_im * r_j_im
        #     message_im = x_j_re * r_j_im + x_j_im * r_j_re
        #     message = torch.cat([message_re, message_im], dim=-1)
        # else:
        #     raise ValueError("Unknown message function `%s`" % self.message_func)

        xj_rel = self.transform(self.p['OPN'], input_j, rel_emb)
        out = torch.einsum('bij,jk->bik', xj_rel, weight)
        out = out if edge_norm is None else out * edge_norm.view(-1, 1)

        # augment messages with the boundary condition
        message = torch.cat([out, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size[0], device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
    # Note: can not do the message and aggregate together. because in the original code, 
    # `relation` is of the shape (num_of_edge_types, dim)
    # def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size,
    #                           # starE added params
    #                           rel_embed, qual_rel_embed, edge_norm, mode, ent_embed,
    #                           qualifier_ent, qualifier_rel, qual_index, source_index):
        
    #     weight = getattr(self, 'w_{}'.format(mode))

    #     if self.p['STATEMENT_LEN'] != 3 and qualifier_rel is not None:
    #         if mode != 'loop':
    #             rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
    #                                                         qualifier_rel, edge_type, qual_index,
    #                                                         qual_rel_embed=qual_rel_embed)
    #         else:
    #             rel_emb = torch.index_select(rel_embed, 0, edge_type)
    #     else:
    #         rel_emb = torch.index_select(rel_embed, 0, edge_type)
        
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        # from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        torch_drug_edge_index = torch.cat([edge_index, edge_type.unsqueeze(0)], dim=0)
        torch_drug_edge_weight = edge_weight
        adjacency = sparse_coo_tensor(torch_drug_edge_index, torch_drug_edge_weight, size=[num_node, num_node, self.num_relation])
        adjacency = adjacency.transpose(0, 1)
        
        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = functional.generalized_rspmm(adjacency, relation, input, sum="add", mul=mul)
            # adjacency = adjacency.transpose(0, 1)
            # sum2 = functional.generalized_rspmm(adjacency, relation, input, sum="add", mul=mul)
            # sum3 = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary) # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def transform(self, option, x, y):
        if option == 'corr':
            trans_embed = ccorr(x, y)
        elif option == 'sub':
            trans_embed = x - y
        elif option == 'mult':
            trans_embed = x * y
        elif option == 'rotate':
            trans_embed = rotate(x, y)
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

    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None, qual_rel_embed=None):
        """
        The update_rel_emb_with_qualifier method performs following functions:

        Input is the secondary COO matrix (QE (qualifier entity), QR (qualifier relation), edge index (Connection to the primary COO))

        Step1 : Embed all the input
            Step1a : Embed the qualifier entity via ent_embed (So QE shape is 33k,1 -> 33k,200)
            Step1b : Embed the qualifier relation via rel_embed (So QR shape is 33k,1 -> 33k,200)
            Step1c : Embed the main statement edge_type via rel_embed (So edge_type shape is 61k,1 -> 61k,200)

        Step2 : Combine qualifier entity emb and qualifier relation emb to create qualifier emb (See self.qual_transform).
            This is generally just summing up. But can be more any pair-wise function that returns one vector for a (qe,qr) vector

        Step3 : Update the edge_type embedding with qualifier information. This uses scatter_add/scatter_mean.


        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [q,w,e',r,t,y,u,i,o,p, .....]        (here q,w,e' .. are of 200 dim each)

        After:
            edge_type          :   [q+(a+b+d),w+(c+e+g),e'+f,......]        (here each element in the list is of 200 dim)


        :param ent_embed: essentially x (28k*200 in case of Jf17k)
        :param rel_embed: essentially relation embedding matrix

        For secondary COO matrix (QE, QR, edge index)
        :param qualifier_ent:  QE
        :param qualifier_rel:  QR
        edge_type:
        :return:

        index select from embedding
        phi operation between qual_ent, qual_rel
        """

        # Step 1: embedding
        if not qual_rel_embed is None:
            qualifier_emb_rel = qual_rel_embed[qualifier_rel]  # TODO: Do not use qual_rel_embed
        else:
            qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        rel_part_emb = rel_embed[edge_type.squeeze()]

        # Step 2: pass it through qual_transform
        qualifier_emb = self.transform(self.p['OPN'], qualifier_emb_ent,
                                       qualifier_emb_rel)

        # Pass it through a aggregate layer
        return self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=self.p['TRIPLE_QUAL_WEIGHT'],
                                        qual_index=qual_index)

    # return qualifier_emb
    # def message(self, x_j, x_i, edge_type, rel_embed, edge_norm, mode, ent_embed=None,
    #             qualifier_ent=None,
    #             qualifier_rel=None, qual_index=None, source_index=None, qual_rel_embed=None):

    #     """

    #     The message method performs following functions

    #     Step1 : get updated relation representation (rel_embed) [edge_type] by aggregating qualifier information (self.update_rel_emb_with_qualifier).
    #     Step2 : Obtain edge message by transforming the node embedding with updated relation embedding (self.rel_transform).
    #     Step3 : Multiply edge embeddings (transform) by weight
    #     Step4 : Return the messages. They will be sent to subjects (1st line in the edge index COO)
    #     Over here the node embedding [the first list in COO matrix] is representing the message which will be sent on each edge


    #     More information about updating relation representation please refer to self.update_rel_emb_with_qualifier

    #     :param x_j: objects of the statements (2nd line in the COO)
    #     :param x_i: subjects of the statements (1st line in the COO)
    #     :param edge_type: relation types
    #     :param rel_embed: embedding matrix of all relations
    #     :param edge_norm:
    #     :param mode: in (direct) / out (inverse) / loop
    #     :param ent_embed: embedding matrix of all entities
    #     :param qualifier_ent:
    #     :param qualifier_rel:
    #     :param qual_index:
    #     :param source_index:
    #     :return:
    #     """
    #     weight = getattr(self, 'w_{}'.format(mode))

    #     if self.p['STATEMENT_LEN'] != 3 and qualifier_rel is not None:
    #         # add code here
    #         if mode != 'loop':
    #             rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
    #                                                          qualifier_rel, edge_type, qual_index,
    #                                                          qual_rel_embed=qual_rel_embed)
    #         else:
    #             rel_emb = torch.index_select(rel_embed, 0, edge_type)
    #     else:
    #         rel_emb = torch.index_select(rel_embed, 0, edge_type)

    #     xj_rel = self.transform(self.p['OPN'], x_j, rel_emb)
    #     out = torch.einsum('ij,jk->ik', xj_rel, weight)
    #     return out if edge_norm is None else out * edge_norm.view(-1, 1)

    # def update(self, aggr_out, mode):
    #     return aggr_out

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """
        Re-normalization trick used by GCN-based architectures without attention.

        Yet another torch scatter functionality. See coalesce_quals for a rough idea.

        row         :      [1,1,2,3,3,4,4,4,4, .....]        (about 61k for Jf17k)
        edge_weight :      [1,1,1,1,1,1,1,1,1,  ....] (same as row. So about 61k for Jf17k)
        deg         :      [2,1,2,4,.....]            (same as num_ent about 28k in case of Jf17k)

        :param edge_index:
        :param num_ent:
        :return:
        """
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # Norm parameter D^{-0.5} *

        return norm

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
        else:
            raise NotImplementedError

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_rels)
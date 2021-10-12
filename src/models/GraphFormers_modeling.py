import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tnlrv3.modeling import TuringNLRv3PreTrainedModel, logger, BertSelfAttention, BertSelfOutput, BertIntermediate, \
    BertOutput, BertLayer, WEIGHTS_NAME, BertEmbeddings, relative_position_bucket
from .tnlrv3.convert_state_dict import get_checkpoint_from_transformer_cache, state_dict_convert

class GraphTuringNLRPreTrainedModel(TuringNLRv3PreTrainedModel):
    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, reuse_position_embedding=None,
            replace_prefix=None, *model_args, **kwargs,
    ):
        model_type = kwargs.pop('model_type', 'tnlrv3')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME),
                                            map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError("Not fined the pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("TNLRv3 does't support the model !")
            raise NotImplementedError()

        config = kwargs["config"]
        state_dict = kwargs["state_dict"]
        # initialize new position embeddings (From Microsoft/UniLM)
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        # initialize new rel_pos weight
        _k = 'bert.rel_pos_bias.weight'
        if _k in state_dict and state_dict[_k].shape[1] != (config.neighbor_type + config.rel_pos_bins):
            logger.info(
                f"rel_pos_bias.weight.shape[1]:{state_dict[_k].shape[1]} != config.bus_num+config.rel_pos_bins:{config.neighbor_type + config.rel_pos_bins}")
            old_rel_pos_bias = state_dict[_k]
            new_rel_pos_bias = torch.cat(
                [old_rel_pos_bias, old_rel_pos_bias[:, -1:].expand(old_rel_pos_bias.size(0), config.neighbor_type)], -1)
            new_rel_pos_bias = nn.Parameter(data=new_rel_pos_bias, requires_grad=True)
            state_dict[_k] = new_rel_pos_bias.data
            del new_rel_pos_bias

        if replace_prefix is not None:
            new_state_dict = {}
            for key in state_dict:
                if key.startswith(replace_prefix):
                    new_state_dict[key[len(replace_prefix):]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            kwargs["state_dict"] = new_state_dict
            del state_dict

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class GraphAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphAggregation, self).__init__(config)
        self.output_attentions = False
        self.mapping_graph = True if config.mapping_graph > 0 else False
        if self.mapping_graph:
            self.selfoutput = BertSelfOutput(config)
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, rel_pos=None, activate_station=None):
        """
        hidden_states[:,0] for the center node, hidden_states[:,1:] for the neighbours
        hidden_states: B SN D
        attention_mask: B 1 1 SN
        rel_pos:B Head_num 1 SN
        """

        query = self.query(hidden_states[:, :1])  # B 1 D
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        # rel_pos=None
        station_embed = self.multi_head_attention(query=query,
                                                  key=key,
                                                  value=value,
                                                  attention_mask=attention_mask,
                                                  rel_pos=rel_pos)[0]  # B 1 D

        if self.mapping_graph:
            attention_output = self.selfoutput(station_embed, hidden_states[:, :1])
            intermediate_output = self.intermediate(attention_output)
            station_embed = self.output(intermediate_output, attention_output)

        station_embed = station_embed.squeeze(1)
        if activate_station is not None:
            station_embed = torch.mul(station_embed, activate_station.unsqueeze(1))

        return station_embed


class GraphBertEncoder(nn.Module):
    def __init__(self, config):
        super(GraphBertEncoder, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.neighbor_type = config.neighbor_type  # config.neighbor_type:  0:No neighbors; 1-2;
        if config.neighbor_type > 0:
            self.graph_attention = GraphAggregation(config=config)
            # self.graph_attention = nn.ModuleList([GraphAggregation(config=config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                node_mask=None,
                node_rel_pos=None,
                rel_pos=None,
                activate_station=None,
                return_last_station_emb=False):
        '''
        Args:
            hidden_states: N L D
            attention_mask: N 1 1 L
            node_mask: B 1 1 subgraph_node_num(SN)
            node_rel_pos: B head_num 1 subgraph_node_num
            rel_pos: N head_num L L
        '''
        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape
        if self.neighbor_type > 0:
            batch_size, _, _, subgraph_node_num = node_mask.shape

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.neighbor_type > 0:
                if i > 0:

                    hidden_states = hidden_states.view(batch_size, subgraph_node_num, seq_length, emb_dim)  # B SN L D
                    cls_emb = hidden_states[:, :, 1].clone()  # B SN D
                    station_emb = self.graph_attention(hidden_states=cls_emb, attention_mask=node_mask,
                                                       rel_pos=node_rel_pos, activate_station=activate_station)  # B D
                    # station_emb = self.graph_attention[i-1](hidden_states=cls_emb, attention_mask=node_mask, rel_pos=node_rel_pos,activate_station=activate_station) #B D

                    # update the station in the query/key
                    hidden_states[:, 0, 0] = station_emb
                    hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)

                    layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, rel_pos=rel_pos)

                else:
                    temp_attention_mask = attention_mask.clone()
                    temp_attention_mask[::subgraph_node_num, :, :, 0] = -10000.0
                    layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask, rel_pos=rel_pos)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, rel_pos=rel_pos)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if return_last_station_emb:
            hidden_states = hidden_states.view(batch_size, subgraph_node_num, seq_length, emb_dim)  # B SN L D
            cls_emb = hidden_states[:, :, 1]  # B SN D
            # station_emb = self.graph_attention[-1](hidden_states=cls_emb, attention_mask=node_mask,rel_pos=node_rel_pos) #B D
            station_emb = self.graph_attention(hidden_states=cls_emb, attention_mask=node_mask,
                                               rel_pos=node_rel_pos)  # B D
            outputs = outputs + (station_emb,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions), (station_emb)


class GraphFormers(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super(GraphFormers, self).__init__(config=config)
        self.config = config
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = GraphBertEncoder(config=config)

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins + self.config.neighbor_type,
                                          config.num_attention_heads,
                                          bias=False)
        else:
            self.rel_pos_bias = None

    def forward(self,
                input_ids,
                attention_mask,
                neighbor_mask=None,
                mask_self_in_graph=False,
                return_last_station_emb=False):
        '''
        Args:
            input_ids: Tensor(N:node_num,L:seq_length)
            attention_mask: Tensor(N,L)
            neighbor_mask: Tensor(B:batch_size, neighbor_num)

        Retures:
            last_hidden_state, (all hidden states), (all attentions), (station_emb)
        '''
        all_nodes_num, seq_length = input_ids.shape
        batch_size, subgraph_node_num = neighbor_mask.shape

        embedding_output, position_ids = self.embeddings(input_ids=input_ids)

        attention_mask = attention_mask.type(embedding_output.dtype)
        neighbor_mask = neighbor_mask.type(embedding_output.dtype)
        node_mask = None
        activate_station = None
        if self.config.neighbor_type > 0:
            station_mask = torch.zeros(all_nodes_num, 1).type(attention_mask.dtype).to(attention_mask.device)  # N 1
            attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 1+L
            # only use the station for selfnode
            attention_mask[::(subgraph_node_num), 0] = 1.0

            if mask_self_in_graph:
                neighbor_mask[:, 0] = 0
                activate_station = torch.sum(neighbor_mask, dim=-1)
                activate_station = activate_station.masked_fill(activate_station > 0, 1)
            node_mask = (1.0 - neighbor_mask[:, None, None, :]) * -10000.0

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        if self.config.rel_pos_bins > 0:
            node_rel_pos = None
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.config.rel_pos_bins,
                                               max_distance=self.config.max_rel_pos)

            if self.config.neighbor_type > 0:
                # rel_pos: (N,L,L) -> (N,1+L,L)
                temp_pos = torch.zeros(all_nodes_num, 1, seq_length, dtype=rel_pos.dtype, device=rel_pos.device)
                rel_pos = torch.cat([temp_pos, rel_pos], dim=1)
                # rel_pos: (N,1+L,L) -> (N,1+L,1+L)
                station_relpos = torch.full((all_nodes_num, seq_length + 1, 1), self.config.rel_pos_bins,
                                            dtype=rel_pos.dtype, device=rel_pos.device)
                rel_pos = torch.cat([station_relpos, rel_pos], dim=-1)

                # node_rel_pos:(B:batch_size, Head_num, neighbor_num+1)
                node_pos = self.config.rel_pos_bins + self.config.neighbor_type - 1
                node_rel_pos = torch.full((batch_size, subgraph_node_num), node_pos, dtype=rel_pos.dtype,
                                          device=rel_pos.device)
                node_rel_pos[:, 0] = 0
                node_rel_pos = F.one_hot(node_rel_pos,
                                         num_classes=self.config.rel_pos_bins + self.config.neighbor_type).type_as(
                    embedding_output)
                node_rel_pos = self.rel_pos_bias(node_rel_pos).permute(0, 2, 1)  # B head_num, neighbor_num
                node_rel_pos = node_rel_pos.unsqueeze(2)  # B head_num 1 neighbor_num

            # rel_pos: (N,Head_num,1+L,1+L)
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins + self.config.neighbor_type).type_as(
                embedding_output)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            # rel_pos[:,:,:,0]=0

        else:
            node_rel_pos = None
            rel_pos = None

        if self.config.neighbor_type > 0:
            # Add station_placeholder
            station_placeholder = torch.zeros(all_nodes_num, 1, embedding_output.size(-1)).type(
                embedding_output.dtype).to(embedding_output.device)
            embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 1+L D

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            node_mask=node_mask,
            node_rel_pos=node_rel_pos,
            rel_pos=rel_pos,
            activate_station=activate_station,
            return_last_station_emb=return_last_station_emb)

        return encoder_outputs

class GraphFormersForNeighborPredict(GraphTuringNLRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = GraphFormers(config)
        if config.graph_transform > 0:
            self.graph_transform = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.init_weights()

    def retrieve_loss(self, q, k):
        score = torch.matmul(q, k.transpose(0, 1))
        loss = F.cross_entropy(score, torch.arange(start=0, end=score.shape[0],
                                                   dtype=torch.long, device=score.device))
        return loss

    def forward(self,
                input_ids_query,
                attention_masks_query,
                mask_query,
                input_ids_key,
                attention_masks_key,
                mask_key,
                neighbor_num,
                mask_self_in_graph=False,
                return_last_station_emb=False):

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num // (neighbor_num + 1)
        neighbor_mask_query = mask_query.view(batch_size, (neighbor_num + 1))
        neighbor_mask_key = mask_key.view(batch_size, (neighbor_num + 1))

        hidden_states_query = self.bert(input_ids_query, attention_masks_query,
                                        neighbor_mask=neighbor_mask_query,
                                        mask_self_in_graph=mask_self_in_graph,
                                        return_last_station_emb=return_last_station_emb
                                        )
        hidden_states_key = self.bert(input_ids_key, attention_masks_key,
                                      neighbor_mask=neighbor_mask_key,
                                      mask_self_in_graph=mask_self_in_graph,
                                      return_last_station_emb=return_last_station_emb
                                      )
        last_hidden_states_query = hidden_states_query[0]
        last_hidden_states_key = hidden_states_key[0]

        # delete the station_placeholder hidden_state:(N,1+L,D)->(N,L,D)
        last_hidden_states_query = last_hidden_states_query[:, 1:]
        last_hidden_states_key = last_hidden_states_key[:, 1:]

        # hidden_state:(N,L,D)->(B,L,D)
        query = last_hidden_states_query[::(neighbor_num + 1)]
        key = last_hidden_states_key[::(neighbor_num + 1)]

        if return_last_station_emb:
            # B D
            last_neighbor_hidden_states_query = hidden_states_query[-1]
            last_neighbor_hidden_states_key = hidden_states_key[-1]

            query = torch.cat([query[:, 0], last_neighbor_hidden_states_query], dim=-1)
            query = self.graph_transform(query)
            key = torch.cat([key[:, 0], last_neighbor_hidden_states_key], dim=-1)
            key = self.graph_transform(key)

        else:
            query = query[:, 0]
            key = key[:, 0]
        neighbor_predict_loss = self.retrieve_loss(query, key)

        return neighbor_predict_loss

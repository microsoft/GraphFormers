import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tnlrv3.modeling import TuringNLRv3PreTrainedModel, TuringNLRv3Model
from src.utils import roc_auc_score, mrr_score, ndcg_score


class GraphSageMaxForNeighborPredict(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = TuringNLRv3Model(config)
        self.graph_transform = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.pooling_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def aggregation(self, neighbor_embed, neighbor_mask):
        neighbor_embed = F.relu(self.pooling_transform(neighbor_embed))
        neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(2) == 0, 0)
        return torch.max(neighbor_embed, dim=-2)[0]

    def graphsage(self, node_embed, node_mask):
        neighbor_embed = node_embed[:, 1:]  # B N D
        neighbor_mask = node_mask[:, 1:]  # B N
        center_embed = node_embed[:, 0]  # B D
        neighbor_embed = self.aggregation(neighbor_embed, neighbor_mask)  # B D
        main_embed = torch.cat([center_embed, neighbor_embed], dim=-1)  # B 2D
        main_embed = self.graph_transform(main_embed)
        main_embed = F.relu(main_embed)
        return main_embed

    def infer(self, input_ids_node_and_neighbors_batch, attention_mask_node_and_neighbors_batch,
              mask_node_and_neighbors_batch):
        B, N, L = input_ids_node_and_neighbors_batch.shape
        D = self.config.hidden_size
        input_ids = input_ids_node_and_neighbors_batch.view(B * N, L)
        attention_mask = attention_mask_node_and_neighbors_batch.view(B * N, L)
        hidden_states = self.bert(input_ids, attention_mask)
        last_hidden_states = hidden_states[0]
        cls_embeddings = last_hidden_states[:, 0].view(B, N, D)  # [B,N,D]
        node_embeddings = self.graphsage(cls_embeddings, mask_node_and_neighbors_batch)
        return node_embeddings

    def test(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
             mask_query_and_neighbors_batch, \
             input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, mask_key_and_neighbors_batch,
             **kwargs):
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                      mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    mask_key_and_neighbors_batch)
        scores = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def forward(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                mask_query_and_neighbors_batch, \
                input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, mask_key_and_neighbors_batch,
                **kwargs):
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                      mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    mask_key_and_neighbors_batch)
        score = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)
        return loss

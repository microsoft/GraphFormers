from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Tuple, Callable, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.dataset import IterableDataset
from transformers import BertTokenizerFast


class DatasetForMatching(IterableDataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: Union[BertTokenizerFast, str] = "bert-base-uncased",
    ):

        self.data_file = open(file_path, "r", encoding="utf-8")
        if isinstance(tokenizer, str):
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    def process(self, input_line):
        # Input file format:
        # Example:
        # A simple algorithm for Boolean operations on polygons|'|Geometric modelling based on simplicial chains|'|Boolean operations on general planar polygons|'|Reentrant polygon clipping|'|Plane-sweep algorithms for intersecting geometric figures|'|A new algorithm for computing Boolean operations on polygons	An analysis and algorithm for polygon clipping|'|Set Membership Classification: A Unified Approach to Geometric Intersection Problems|'|Reentrant polygon clipping|'|Hidden surface removal using polygon area sorting|'|Polygon comparison using a graph representation|'|A New Concept and Method for Line Clipping
        # Balanced Multifilter Banks for Multiple Description Coding|'|Balanced multiwavelets|'|On minimal lattice factorizations of symmetric-antisymmetric multifilterbanks|'|High-order balanced multiwavelets: theory, factorization, and design|'|Single-Trial Multiwavelet Coherence in Application to Neurophysiological Time Series|'|The application of multiwavelet filterbanks to image processing	Armlets and balanced multiwavelets: flipping filter construction|'|Multiwavelet prefilters. II. Optimal orthogonal prefilters|'|Regularity of multiwavelets|'|Balanced GHM-like multiscaling functions|'|A new prefilter design for discrete multiwavelet transforms|'|Balanced multiwavelets with short filters

        query_and_neighbors, key_and_neighbors = input_line.strip('\n').split('\t')[:2]
        query_and_neighbors = query_and_neighbors.split('|\'|')
        key_and_neighbors = key_and_neighbors.split('|\'|')
        tokens_query_and_neighbors = self.tokenizer.batch_encode_plus(query_and_neighbors, add_special_tokens=False)[
            'input_ids']
        tokens_key_and_neighbors = self.tokenizer.batch_encode_plus(key_and_neighbors, add_special_tokens=False)[
            'input_ids']

        return tokens_query_and_neighbors, tokens_key_and_neighbors

    def __iter__(self):
        for line in self.data_file:
            yield self.process(line)


@dataclass
class DataCollatorForMatching:
    mlm: bool
    neighbor_num: int
    token_length: int
    tokenizer: Union[BertTokenizerFast, str] = "bert-base-uncased"
    mlm_probability: float = 0.15
    random_seed: int = 42

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer)
        self.random_state = np.random.RandomState(seed=self.random_seed)

    def __call__(self, samples: List[List[List[List[int]]]]) -> Dict[str, torch.Tensor]:
        input_ids_query_and_neighbors_batch = []
        attention_mask_query_and_neighbors_batch = []
        mask_query_and_neighbors_batch = []
        input_ids_key_and_neighbors_batch = []
        attention_mask_key_and_neighbors_batch = []
        mask_key_and_neighbors_batch = []
        for i, sample in (enumerate(samples)):
            input_ids_query_and_neighbors, attention_mask_query_and_neighbors, mask_query_and_neighbors, \
            input_ids_key_and_neighbors, attention_mask_key_and_neighbors, mask_key_and_neighbors = self.create_training_sample(
                sample)

            input_ids_query_and_neighbors_batch.append(input_ids_query_and_neighbors)
            attention_mask_query_and_neighbors_batch.append(attention_mask_query_and_neighbors)
            mask_query_and_neighbors_batch.append(mask_query_and_neighbors)

            input_ids_key_and_neighbors_batch.append(input_ids_key_and_neighbors)
            attention_mask_key_and_neighbors_batch.append(attention_mask_key_and_neighbors)
            mask_key_and_neighbors_batch.append(mask_key_and_neighbors)

        if self.mlm:
            input_ids_query_and_neighbors_batch, mlm_labels_query_batch = self.mask_tokens(
                self._tensorize_batch(input_ids_query_and_neighbors_batch, self.tokenizer.pad_token_id),
                self.tokenizer.mask_token_id)
            input_ids_key_and_neighbors_batch, mlm_labels_key_batch = self.mask_tokens(
                self._tensorize_batch(input_ids_key_and_neighbors_batch, self.tokenizer.pad_token_id),
                self.tokenizer.mask_token_id)
        else:
            input_ids_query_and_neighbors_batch = self._tensorize_batch(input_ids_query_and_neighbors_batch,
                                                                        self.tokenizer.pad_token_id)
            input_ids_key_and_neighbors_batch = self._tensorize_batch(input_ids_key_and_neighbors_batch,
                                                                      self.tokenizer.pad_token_id)
        attention_mask_query_and_neighbors_batch = self._tensorize_batch(attention_mask_query_and_neighbors_batch, 0)
        attention_mask_key_and_neighbors_batch = self._tensorize_batch(attention_mask_key_and_neighbors_batch, 0)
        mask_query_and_neighbors_batch = self._tensorize_batch(mask_query_and_neighbors_batch, 0)
        mask_key_and_neighbors_batch = self._tensorize_batch(mask_key_and_neighbors_batch, 0)

        return {
            "input_ids_query_and_neighbors_batch": input_ids_query_and_neighbors_batch,
            "attention_mask_query_and_neighbors_batch": attention_mask_query_and_neighbors_batch,
            "mlm_labels_query_batch": mlm_labels_query_batch if self.mlm else None,
            "mask_query_and_neighbors_batch": mask_query_and_neighbors_batch,
            "input_ids_key_and_neighbors_batch": input_ids_key_and_neighbors_batch,
            "attention_mask_key_and_neighbors_batch": attention_mask_key_and_neighbors_batch,
            "mlm_labels_key_batch": mlm_labels_key_batch if self.mlm else None,
            "mask_key_and_neighbors_batch": mask_key_and_neighbors_batch,
        }

    def _tensorize_batch(self, sequences: Union[List[torch.Tensor], List[List[torch.Tensor]]],
                         padding_value) -> torch.Tensor:
        if len(sequences[0].size()) == 1:
            max_len_1 = max([s.size(0) for s in sequences])
            out_dims = (len(sequences), max_len_1)
            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length_1 = tensor.size(0)
                out_tensor[i, :length_1] = tensor
            return out_tensor
        elif len(sequences[0].size()) == 2:
            max_len_1 = max([s.size(0) for s in sequences])
            max_len_2 = max([s.size(1) for s in sequences])
            out_dims = (len(sequences), max_len_1, max_len_2)
            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length_1 = tensor.size(0)
                length_2 = tensor.size(1)
                out_tensor[i, :length_1, :length_2] = tensor
            return out_tensor
        else:
            raise

    def create_training_sample(self, sample: List[List[List[int]]]):

        def process_node_and_neighbors(tokens_node_and_neighbors):
            max_num_tokens = self.token_length - self.tokenizer.num_special_tokens_to_add(pair=False)
            input_ids_node_and_neighbors, attention_mask_node_and_neighbors, mask_node_and_neighbors = [], [], []
            for i, tokens in enumerate(tokens_node_and_neighbors):
                if i > self.neighbor_num: break
                input_ids_node_and_neighbors.append(
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokens[:max_num_tokens])))
                attention_mask_node_and_neighbors.append(torch.tensor([1] * len(input_ids_node_and_neighbors[-1])))
                if len(tokens) == 0:
                    mask_node_and_neighbors.append(torch.tensor(0))
                else:
                    mask_node_and_neighbors.append(torch.tensor(1))
            input_ids_node_and_neighbors = self._tensorize_batch(input_ids_node_and_neighbors,
                                                                 self.tokenizer.pad_token_id)
            attention_mask_node_and_neighbors = self._tensorize_batch(attention_mask_node_and_neighbors, 0)
            mask_node_and_neighbors = torch.stack(mask_node_and_neighbors)
            return input_ids_node_and_neighbors, attention_mask_node_and_neighbors, mask_node_and_neighbors

        tokens_query_and_neighbors, tokens_key_and_neighbors = sample
        input_ids_query_and_neighbors, attention_mask_query_and_neighbors, mask_query_and_neighbors = process_node_and_neighbors(
            tokens_query_and_neighbors)
        input_ids_key_and_neighbors, attention_mask_key_and_neighbors, mask_key_and_neighbors = process_node_and_neighbors(
            tokens_key_and_neighbors)

        return input_ids_query_and_neighbors, attention_mask_query_and_neighbors, mask_query_and_neighbors, \
               input_ids_key_and_neighbors, attention_mask_key_and_neighbors, mask_key_and_neighbors

    def mask_tokens(self, inputs_origin: torch.Tensor, mask_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        inputs = inputs_origin.clone()
        labels = torch.zeros((inputs.shape[0], inputs.shape[2]), dtype=torch.long) - 100
        for i in range(len(inputs_origin)):
            input_origin = inputs_origin[i][0]
            input = inputs[i][0]
            mask_num, valid_length = 0, 0
            start_indexes = []
            for index, x in enumerate(input_origin):
                if int(x) not in self.tokenizer.all_special_ids:
                    valid_length += 1
                    start_indexes.append(index)
                    labels[i][index] = -99
            self.random_state.shuffle(start_indexes)
            if valid_length > 0:
                while mask_num / valid_length < self.mlm_probability:
                    start_index = start_indexes.pop()
                    span_length = 1e9
                    while span_length > 10: span_length = np.random.geometric(0.2)
                    for j in range(start_index, min(start_index + span_length, len(input_origin))):
                        if labels[i][j] != -99: continue
                        labels[i][j] = input_origin[j].clone()
                        rand = self.random_state.random()
                        if rand < 0.8:
                            input[j] = mask_id
                        elif rand < 0.9:
                            input[j] = self.random_state.randint(0, self.tokenizer.vocab_size - 1)
                        mask_num += 1
                        if mask_num / valid_length >= self.mlm_probability:
                            break
            labels[i] = torch.masked_fill(labels[i], labels[i] < 0, -100)
        return inputs, labels


@dataclass
class MultiProcessDataLoader:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    local_rank: int
    world_size: int
    global_end: Any
    blocking: bool = False
    drop_last: bool = True

    def _start(self):
        self.local_end = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        for batch in self._generate_batch():
            self.outputs.put(batch)
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            if i % self.world_size != self.local_rank: continue
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end = True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        dist.barrier()
        while self.aval_count == 0:
            if self.local_end or self.global_end.value:
                self.global_end.value = True
                break
        dist.barrier()
        if self.global_end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch


@dataclass
class SingleProcessDataLoader:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    blocking: bool = False
    drop_last: bool = True

    def _start(self):
        self.local_end = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        for batch in self._generate_batch():
            self.outputs.put(batch)
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end = True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        while self.aval_count == 0:
            if self.local_end: raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch

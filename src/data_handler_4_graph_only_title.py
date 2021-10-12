import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
from torch.utils.data.dataset import IterableDataset
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import json
import os
from queue import Queue
from filelock import FileLock

from transformers.utils import logging
from transformers import BertTokenizerFast
from concurrent.futures import ThreadPoolExecutor

logger = logging.get_logger(__name__)


class DatasetForMatching(IterableDataset):
    def __init__(
            self,
            tokenizer: BertTokenizerFast,
            file_path: str,
            neighbor_num:int,
            overwrite_cache=False,
            tokenizing_batch_size=32768
    ):
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_{}_{}".format(
                tokenizer.__class__.__name__,
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # Example:
        # A simple algorithm for Boolean operations on polygons|'|Geometric modelling based on simplicial chains|'|Boolean operations on general planar polygons|'|Reentrant polygon clipping|'|Plane-sweep algorithms for intersecting geometric figures|'|A new algorithm for computing Boolean operations on polygons	An analysis and algorithm for polygon clipping|'|Set Membership Classification: A Unified Approach to Geometric Intersection Problems|'|Reentrant polygon clipping|'|Hidden surface removal using polygon area sorting|'|Polygon comparison using a graph representation|'|A New Concept and Method for Line Clipping
        # Balanced Multifilter Banks for Multiple Description Coding|'|Balanced multiwavelets|'|On minimal lattice factorizations of symmetric-antisymmetric multifilterbanks|'|High-order balanced multiwavelets: theory, factorization, and design|'|Single-Trial Multiwavelet Coherence in Application to Neurophysiological Time Series|'|The application of multiwavelet filterbanks to image processing	Armlets and balanced multiwavelets: flipping filter construction|'|Multiwavelet prefilters. II. Optimal orthogonal prefilters|'|Regularity of multiwavelets|'|Balanced GHM-like multiscaling functions|'|A new prefilter design for discrete multiwavelet transforms|'|Balanced multiwavelets with short filters

        with FileLock(lock_path):
            if os.path.exists(cached_features_file + ".finish") and not overwrite_cache:
                self.data_file = open(cached_features_file, "r", encoding="utf-8")
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                batch_query, batch_key = [], []
                with open(file_path, encoding="utf-8") as f, open(cached_features_file, "w", encoding="utf-8")as fout:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        query_and_nn,key_and_nn=line.strip('\n').split('\t')[:2]
                        for query in query_and_nn.split("|\'|"):
                            query=query.strip()
                            if not query:
                                batch_query.append("")
                            else:
                                batch_query.append(query)

                        for key in key_and_nn.split("|\'|"):
                            key=key.strip()
                            if not key:
                                batch_key.append("")
                            else:
                                batch_key.append(key)
                        if len(batch_query) >= tokenizing_batch_size:
                            tokenized_result_query = tokenizer.batch_encode_plus(batch_query,add_special_tokens=False)
                            tokenized_result_key = tokenizer.batch_encode_plus(batch_key,add_special_tokens=False)
                            samples=[[],[]]
                            for j,(tokens_query, tokens_key) in enumerate(zip(tokenized_result_query['input_ids'],
                                                                           tokenized_result_key['input_ids'])):
                                samples[0].append(tokens_query)
                                samples[1].append(tokens_key)
                                if j%(neighbor_num+1)==neighbor_num:
                                    fout.write(json.dumps(samples)+'\n')
                                    samples=[[],[]]
                            batch_query, batch_key = [], []

                    if len(batch_query) > 0:
                        tokenized_result_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
                        tokenized_result_key = tokenizer.batch_encode_plus(batch_key, add_special_tokens=False)
                        samples = [[], []]
                        for j, (tokens_query, tokens_key) in enumerate(zip(tokenized_result_query['input_ids'],
                                                                           tokenized_result_key['input_ids'])):
                            samples[0].append(tokens_query)
                            samples[1].append(tokens_key)
                            if j % (neighbor_num + 1) == neighbor_num:
                                fout.write(json.dumps(samples) + '\n')
                                samples = [[], []]
                        batch_query, batch_key = [], []
                    logger.info(f"Finish creating")
                with open(cached_features_file + ".finish", "w", encoding="utf-8"):
                    pass
                self.data_file = open(cached_features_file, "r", encoding="utf-8")

    def __iter__(self):
        for line in self.data_file:
            tokens_title = json.loads(line)
            yield tokens_title


@dataclass
class DataCollatorForMatching:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: BertTokenizerFast
    mlm: bool
    neighbor_num:int
    neighbor_mask:bool
    block_size: int
    mlm_probability: float = 0.15

    def __call__(self, samples: List[List[List[List[int]]]]) -> Dict[str, torch.Tensor]:
        input_id_queries=[]
        attention_mask_queries=[]
        mask_queries=[]
        input_id_keys=[]
        attention_mask_keys=[]
        mask_keys=[]
        for i, sample in (enumerate(samples)):
            input_id_queries_and_nn,attention_mask_queries_and_nn,mask_query,input_id_keys_and_nn,attention_mask_keys_and_nn,mask_key = self.create_training_sample(sample)
            input_id_queries.extend(input_id_queries_and_nn)
            attention_mask_queries.extend(attention_mask_queries_and_nn)
            mask_queries.extend(mask_query)
            input_id_keys.extend(input_id_keys_and_nn)
            attention_mask_keys.extend(attention_mask_keys_and_nn)
            mask_keys.extend(mask_key)
        if self.mlm:
            input_id_queries, mlm_labels_queries = self.mask_tokens(self._tensorize_batch(input_id_queries, self.tokenizer.pad_token_id), self.tokenizer.mask_token_id)
            input_id_keys, mlm_labels_keys = self.mask_tokens(self._tensorize_batch(input_id_keys, self.tokenizer.pad_token_id), self.tokenizer.mask_token_id)
        else:
            input_id_queries = self._tensorize_batch(input_id_queries, self.tokenizer.pad_token_id)
            input_id_keys = self._tensorize_batch(input_id_keys, self.tokenizer.pad_token_id)
        mask_queries=torch.tensor(mask_queries)
        mask_keys=torch.tensor(mask_keys)
        return {
            "input_id_query": input_id_queries,
            "attention_masks_query": self._tensorize_batch(attention_mask_queries, 0),
            "masked_lm_labels_query": mlm_labels_queries if self.mlm else None,
            "mask_query":mask_queries,
            "input_id_key": input_id_keys,
            "attention_masks_key": self._tensorize_batch(attention_mask_keys, 0),
            "masked_lm_labels_key": mlm_labels_keys if self.mlm else None,
            "mask_key":mask_keys,
        }

    def _tensorize_batch(self, examples: List[torch.Tensor], padding_value) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=padding_value)

    def create_training_sample(self, sample: List[List[List[int]]]):
        """Creates a training sample from the tokens of a title."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        token_queries,token_keys = sample

        mask_queries, mask_keys = [], []
        query_neighbor_list=[]
        key_neighbor_list=[]
        for i, (token_query, token_key) in enumerate(zip(token_queries, token_keys)):
            if len(token_query)==0:
                mask_queries.append(torch.tensor(0))
            else:
                if i!=0: query_neighbor_list.append(i)
                mask_queries.append(torch.tensor(1))
            if len(token_key)==0:
                mask_keys.append(torch.tensor(0))
            else:
                if i!=0: key_neighbor_list.append(i)
                mask_keys.append(torch.tensor(1))

        if self.neighbor_mask:
            if np.random.random() < 0.5:
                mask_query_neighbor_num = min(np.random.randint(1, self.neighbor_num),len(query_neighbor_list))
            else:
                mask_query_neighbor_num = 0
            if np.random.random() < 0.5:
                mask_key_neighbor_num = min(np.random.randint(1, self.neighbor_num),len(key_neighbor_list))
            else:
                mask_key_neighbor_num = 0

            mask_query_set = set(
                np.random.choice(query_neighbor_list, mask_query_neighbor_num, replace=False))
            mask_key_set = set(
                np.random.choice(key_neighbor_list, mask_key_neighbor_num, replace=False))

        input_id_queries,input_id_keys,attention_mask_queries,attention_mask_keys=[],[],[],[]
        for i,(token_query,token_key) in enumerate(zip(token_queries,token_keys)):
            input_id_queries.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(token_query[:max_num_tokens])))
            input_id_keys.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(token_key[:max_num_tokens])))
            attention_mask_queries.append(torch.tensor([1]*len(input_id_queries[-1])))
            attention_mask_keys.append(torch.tensor([1]*len(input_id_keys[-1])))
            if self.neighbor_mask:
                if i in mask_query_set:
                    mask_queries[i]=torch.tensor(0)
                if i in mask_key_set:
                    mask_keys[i]=torch.tensor(0)

        return input_id_queries,attention_mask_queries,mask_queries,input_id_keys,attention_mask_keys,mask_keys

    def mask_tokens(self, inputs_origin: torch.Tensor, mask_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        inputs = inputs_origin.clone()
        labels = torch.zeros((inputs.shape[0]//(self.neighbor_num+1),inputs.shape[1]),dtype=torch.long)-100
        num=0
        for i, input_origin in enumerate(inputs_origin):
            if i%(self.neighbor_num+1)!=0:continue
            mask_num, valid_length = 0, 0
            start_indexes=[]
            for index, x in enumerate(input_origin):
                if int(x) not in self.tokenizer.all_special_ids:
                    valid_length += 1
                    start_indexes.append(index)
                    labels[num][index] = -99
            random.shuffle(start_indexes)
            if valid_length>0:
                while mask_num / valid_length < self.mlm_probability:
                    start_index = start_indexes.pop()
                    span_length = 1e9
                    while span_length > 10: span_length = np.random.geometric(0.2)
                    for j in range(start_index, min(start_index+span_length,len(input_origin))):
                        if labels[num][j] != -99: continue
                        labels[num][j] = input_origin[j].clone()
                        rand=np.random.random()
                        if rand<0.8:
                            inputs[i][j] = mask_id
                        elif rand<0.9:
                            inputs[i][j]=np.random.randint(0,self.tokenizer.vocab_size-1)
                        mask_num += 1
                        if mask_num / valid_length >= self.mlm_probability:
                            break
            labels[num] = torch.masked_fill(labels[num], labels[num] < 0, -100)
            num+=1
        return inputs, labels


@dataclass
class MultiProcessDataLoaderForMatching:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    local_rank: int
    world_size: int
    global_end: Any
    blocking: bool=False
    drop_last: bool = True

    def _start(self):
        self.local_end=False
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
            if len(batch)>=self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end=True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        dist.barrier()
        while self.aval_count == 0:
            if self.local_end or self.global_end.value:
                self.global_end.value=True
                break
        dist.barrier()
        if self.global_end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch

@dataclass
class SingleProcessDataLoaderForMatching:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    blocking: bool=False
    drop_last: bool = True

    def _start(self):
        self.local_end=False
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
            if len(batch)>=self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end=True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        while self.aval_count==0:
            if self.local_end:raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch
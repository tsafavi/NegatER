import os
import json
import torch

import numpy as np

from torch.utils.data import DataLoader
from collections import defaultdict

from job.job import Job
from dataset.knowledge_base import KnowledgeBase
from dataset.tokenize import BaseTripleTokenizer
from job.nearest_neighbors import NearestNeighborsSearchJob


class NegativeKnowledgeBaseCandidates(KnowledgeBase):
    """A dataset of negative knowledge base candidate triples (tokenized)"""

    def __init__(self, config_path, triples):
        super().__init__(
            config_path,
            triples=triples,
            job_key="negater.fine_tuned_model",
            model_key="negater.fine_tuned_model",
        )

        self.tokenizer = BaseTripleTokenizer.create_for(self)

    def __getitem__(self, i):
        return self.tokenizer.encode(self.triples[i], triple_label=1.0)


class NegativeGenerationJob(Job):
    """Base class for generating candidate negative triples"""

    def __init__(self, config_path):
        super().__init__(config_path)

        self.dataset = KnowledgeBase(self.config_path)
        self.indexer = NearestNeighborsSearchJob(self.config_path)

    @torch.no_grad()
    def run(self):
        raise NotImplementedError

    def _generate_corruptions(self):
        # Load nearest-neighbors index from file or build it
        if self.get_option("negater.index.build"):
            nearest_neighbors = self.indexer.run()
        else:
            nearest_neighbors = defaultdict(
                list, json.load(open(self.indexer.filename))
            )

            self.log(f"Loaded nearest-neighbors index from {self.indexer.filename}")

        self.log("Generating nearest-neighbor corruptions...")

        forward = defaultdict(lambda: defaultdict(set))
        backward = defaultdict(lambda: defaultdict(set))
        head_slots, tail_slots = defaultdict(set), defaultdict(set)

        for i in range(len(self.dataset)):
            head, relation, tail = self.dataset[i]

            forward[head][relation].add(tail)
            backward[tail][relation].add(head)

            head_slots[relation].add(head)
            tail_slots[relation].add(tail)

        corruptions = []
        for i in range(len(self.dataset)):
            head, relation, tail = self.dataset[i]

            head_filter = backward[tail][relation].union({head, tail})
            tail_filter = forward[head][relation].union({head, tail})

            valid_heads = head_slots[relation].difference(head_filter)
            valid_tails = tail_slots[relation].difference(tail_filter)

            heads = set(nearest_neighbors[head]).intersection(valid_heads)
            tails = set(nearest_neighbors[tail]).intersection(valid_tails)

            corruptions.extend([(head_, relation, tail) for head_ in heads])
            corruptions.extend([(head, relation, tail_) for tail_ in tails])

        # Randomly shuffle the triples, and replace the original positives
        idx = np.arange(len(corruptions))
        np.random.shuffle(idx)

        self.log(f"Generated {len(corruptions)} nearest-neighbor corruptions")
        return np.asarray(corruptions)[idx]


class NegativeGenerationFullJob(NegativeGenerationJob):
    """Generate negatives and return them in a single dataset"""

    def __init__(self, config_path):
        super().__init__(config_path)

    @torch.no_grad()
    def run(self):
        corruptions = self._generate_corruptions()
        dataset = NegativeKnowledgeBaseCandidates(self.config_path, corruptions)
        return dataset


class NegativeGenerationByRelationJob(NegativeGenerationJob):
    """Generate negatives and return them in separate datasets by relation"""

    def __init__(self, config_path):
        super().__init__(config_path)

    @torch.no_grad()
    def run(self):
        corruptions = self._generate_corruptions()

        datasets = {}
        for relation in self.dataset.relations:
            corruptions_relation = corruptions[corruptions[:, 1] == relation]
            datasets[relation] = NegativeKnowledgeBaseCandidates(
                self.config_path,
                corruptions_relation,
            )

        return datasets

import random
import json

import numpy as np
import pandas as pd

from collections import defaultdict
from scipy import sparse

from config import Configurable


class BaseNegativeSampler(Configurable):
    """Base class for sampling negatives in a commonsense KB"""

    def __init__(self, dataset):
        """The constructor shouldn't be called directly; use the static create_for
        factory method instead"""
        super().__init__(
            dataset.config_path, job_key=f"{dataset.job_key}.negative_sampling"
        )

        self.dataset = dataset
        self.nsamp = np.asarray(
            [
                self.get_option(f"{self.job_key}.nsamp.{samp}", default=0)
                for samp in ("head", "relation", "tail")
            ]
        )

        self.nsampling_thresholds = np.cumsum(self.nsamp) + 1

    def __len__(self):
        """Returns the number of samples to be generated per epoch"""
        return len(self.dataset.triples) * (np.sum(self.nsamp))

    @staticmethod
    def create_for(dataset):
        """Static factory method"""
        sampler_type = dataset.get_option(
            f"{dataset.job_key}.negative_sampling.sampler"
        ).lower()

        if sampler_type == "uniform":
            return UniformRandomSampler(dataset)
        elif sampler_type == "slots":
            return SlotBasedRandomSampler(dataset)
        elif sampler_type == "simple-neg":
            return SimpleNegationSampler(dataset)
        elif sampler_type == "antonyms":
            return AntonymSubstitutionSampler(dataset)
        elif sampler_type == "sans":
            return SANSSampler(dataset)
        return PrecomputedNegativeSampler(dataset)

    def sample(self, index, head, relation, tail):
        raise NotImplementedError


class UniformRandomSampler(BaseNegativeSampler):
    """Sample negatives with uniform random corruption"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def sample(self, index, head, relation, tail):
        """Randomly corrupt the head, relation, or tail of the triple"""
        sample_idx = index // len(self.dataset.triples)

        if sample_idx < self.nsampling_thresholds[0]:
            head = random.choice(self.dataset.entities)
        elif sample_idx < self.nsampling_thresholds[1]:
            relation = random.choice(self.dataset.relations)
        else:
            tail = random.choice(self.dataset.entities)

        return head, relation, tail


class SlotBasedRandomSampler(BaseNegativeSampler):
    """Corrupt the head or tail by sampling from previously seen head/tail phrases
    for this relation, or corrupt the relation at random"""

    def __init__(self, dataset):
        super().__init__(dataset)

        # Map each relation to possible head and tail phrases from dataset
        head_slots, tail_slots = defaultdict(set), defaultdict(set)

        for (head, relation, tail) in self.dataset.triples:
            head_slots[relation].add(head)
            tail_slots[relation].add(tail)

        self.head_slots = {
            relation: list(heads) for relation, heads in head_slots.items()
        }
        self.tail_slots = {
            relation: list(tails) for relation, tails in tail_slots.items()
        }

    def sample(self, index, head, relation, tail):
        sample_idx = index // len(self.dataset.triples)

        if sample_idx < self.nsampling_thresholds[0]:
            head = random.choice(self.head_slots[relation])
        elif sample_idx < self.nsampling_thresholds[1]:
            relation = random.choice(self.relations)
        else:
            tail = random.choice(self.tail_slots[relation])

        return head, relation, tail


class SimpleNegationSampler(BaseNegativeSampler):
    """Corrupt the head or tail phrase by prepending a "not" token to the phrase, or
    corrupt the relation at random"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def sample(self, index, head, relation, tail):
        sample_idx = index // len(self.dataset.triples)

        if sample_idx < self.nsampling_thresholds[0]:
            head = "not " + head
        elif sample_idx < self.nsampling_thresholds[1]:
            relation = random.choice(self.relations)
        else:
            tail = "not " + tail

        return head, relation, tail


class AntonymSubstitutionSampler(BaseNegativeSampler):
    """Corrupt the head/tail phrase by replacing a token with its antonym if available,
    or else fall back to a base sampler"""

    def __init__(self, dataset):
        super().__init__(dataset)

        # Default to base sampler if antonym isn't found
        self.base_sampler = SlotBasedRandomSampler(dataset)
        self.job_key = f"{self.job_key}.antonyms"

        # Load pre-computed antonymy mappings
        mapping_file = self.get_option(f"{self.job_key}.mapping_file")
        token_file = self.get_option(f"{self.job_key}.token_file")
        tag_file = self.get_option(f"{self.job_key}.tag_file")

        with open(mapping_file) as f:
            self.antonyms = json.load(f)
        self.log(f"Loaded antonym mapping from {mapping_file}")

        with open(token_file) as f:
            self.tokenized_phrases = json.load(f)
        self.log(f"Loaded phrase tokenization from {token_file}")

        with open(tag_file) as f:
            self.tagged_phrases = json.load(f)
        self.log(f"Loaded phrase tags from {tag_file}")

    def sample(self, index, head, relation, tail):
        sample_idx = index // len(self.dataset.triples)

        if sample_idx < self.nsampling_thresholds[0]:
            head_split = [token for token in self.tokenized_phrases[head]]
            token_idx = self.tagged_phrases[head]
            replace_token = head_split[token_idx]

            if replace_token not in self.antonyms:
                return self.base_sampler.sample(index, head, relation, tail)
            else:
                head_split[token_idx] = random.choice(self.antonyms[replace_token])
                head = " ".join(head_split)
        elif sample_idx < self.nsampling_thresholds[1]:
            relation = random.choice(self.relations)
        else:
            tail_split = [token for token in self.tokenized_phrases[tail]]
            token_idx = self.tagged_phrases[tail]
            replace_token = tail_split[token_idx]

            if replace_token not in self.antonyms:
                return self.base_sampler.sample(index, head, relation, tail)
            else:
                tail_split[token_idx] = random.choice(self.antonyms[replace_token])
                tail = " ".join(tail_split)

        return head, relation, tail


class SANSSampler(BaseNegativeSampler):
    """Corrupt head/tail phrases by sampling from a pre-computed k-hop matrix obtained
    with SANS: https://github.com/kahrabian/SANS"""

    def __init__(self, dataset):
        super().__init__(dataset)

        self.job_key = f"{self.job_key}.sans"
        kmat_file = self.get_option(f"{self.job_key}.kmat_file")
        self.k_neighbors = sparse.load_npz(kmat_file)

        self.entity_ids = {entity: i for i, entity in enumerate(self.dataset.entities)}

    def sample(self, index, head, relation, tail):
        sample_idx = index // len(self.dataset.triples)

        if sample_idx < self.nsampling_thresholds[0]:
            tail_idx = self.entity_ids[tail]
            neighbors = self.k_neighbors[tail_idx].indices
            head = self.dataset.entities[random.choice(neighbors)]
        elif sample_idx < self.nsampling_thresholds[1]:
            relation = random.choice(self.relations)
        else:
            head_idx = self.entity_ids[head]
            neighbors = self.k_neighbors[head_idx].indices
            tail = self.dataset.entities[random.choice(neighbors)]

        return (head, relation, tail)


class PrecomputedNegativeSampler(BaseNegativeSampler):
    """Sample negatives from a file of pre-generated negative samples"""

    def __init__(self, dataset):
        super().__init__(dataset)

        # Read the hard negative examples from a file
        negative_file = self.get_option(f"{self.job_key}.precomputed.negative_file")
        negatives = pd.read_csv(negative_file, sep="\t")[
            ["head", "relation", "tail"]
        ]
        negatives["head"] = negatives["head"].replace({np.nan: "none"})
        negatives["tail"] = negatives["tail"].replace({np.nan: "none"})

        self.negatives = negatives.values
        self.log(f"Loaded {len(self.negatives)} training negatives from {negative_file}")

        self.n_negatives_per_epoch = min(
            len(self),
            len(self.negatives) // self.get_option(f"{dataset.job_key}.n_epochs"),
        )  # how many pre-computed negative statements can we reserve per epoch?

        self.negative_index = 0  # global index for sampling

    def sample(self, index, head, relation, tail):
        negative = self.negatives[self.negative_index]
        self.negative_index += 1

        if self.negative_index >= len(self.negatives):  # restart index
            self.negative_index %= len(self.negatives)

        return negative

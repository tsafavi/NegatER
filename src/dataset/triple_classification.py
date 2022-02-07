import os
import torch

import numpy as np

from tqdm import tqdm

from dataset.knowledge_base import KnowledgeBase
from dataset.tokenize import BaseTripleTokenizer
from dataset.negative_sampler import BaseNegativeSampler
from utils import save_checkpoint, load_checkpoint


class TripleClassificationDataset(KnowledgeBase):
    """A tokenized version of a knowledge base"""

    def __init__(self, config_path):
        super().__init__(config_path, job_key="fine_tune", model_key="fine_tune.model")

        self.tokenizer = BaseTripleTokenizer.create_for(self)
        self.pretokenize_triples = self.get_option("dataset.pretokenize_triples")
        if self.pretokenize_triples:
            self._pretokenize()

    def _split(self):
        raise NotImplementedError

    def _pretokenize(self):
        """Pre-tokenize the data split once and save the tensors to files, or
        load the pretokenized files"""
        triple_token_path = os.path.join(
            self.dataset_path, self.get_option("dataset.triples_tokenized_path")
        )
        split = self._split()
        tokens_filename = os.path.join(triple_token_path, f"{split}.pt")

        if os.path.exists(tokens_filename):
            self.tokenized_kb = load_checkpoint(tokens_filename, device=self.device)

            self.log(f"Loaded tokenized {split} split from {tokens_filename}")
        else:
            self.tokenized_kb = []
            for triple in tqdm(self.triples, desc=f"Tokenizing {split} split"):
                self.tokenized_kb.append(self.tokenizer.encode(triple))

            if not os.path.isdir(triple_token_path):
                os.makedirs(triple_token_path)

            save_checkpoint(self.tokenized_kb, tokens_filename)
            self.log(f"Saved tokenized {split} split to {tokens_filename}")


class FineTuningDataset(TripleClassificationDataset):
    """Training split for triple classification, including negative samples"""

    def __init__(self, config_path):
        super().__init__(config_path)

        self.job_key = f"{self.job_key}.train"
        self.negative_sampler = BaseNegativeSampler.create_for(self)

    def __len__(self):
        return len(self.triples) + len(self.negative_sampler)  # account for negatives

    def __getitem__(self, i):
        """Get a positive or negative triple"""
        triple = self.triples[i % len(self.triples)]
        label = int(i < len(self.triples))

        if not label:  # sample a negative - not pretokenized
            tokenizer_input = self.negative_sampler.sample(i, *triple)
        elif self.pretokenize_triples:  # lookup positive pre-tokenized triple
            tokenizer_input = self.tokenized_kb[i]
        else:  # just feed the triple directly to the tokenizer
            tokenizer_input = triple

        return self.tokenizer.encode(
            tokenizer_input,
            triple_label=label,
            is_pretokenized=self.pretokenize_triples and label,
        )

    def _load_triples(self):
        self.triples = self._load_split("train")

    def _split(self):
        return "train"


class EvaluationDataset(TripleClassificationDataset):
    def __init__(self, config_path):
        super().__init__(config_path)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        tokenizer_input = (
            self.tokenized_kb[i] if self.pretokenize_triples else self.triples[i]
        )

        return self.tokenizer.encode(
            tokenizer_input,
            triple_label=self.labels[i],
            is_pretokenized=self.pretokenize_triples,
        )

    def _load_triples(self):
        """Map all triple IDs (pos and neg) to text"""
        triples, labels = [], []

        for label in (True, False):
            split_filename = self._split()
            if not label:
                split_filename += "_negatives"

            data = self._load_split(split_filename)
            triples.append(data)
            labels += [int(label)] * len(data)

        self.triples = np.concatenate(triples, axis=0)
        self.labels = labels


class ValidationDataset(EvaluationDataset):
    """Validation split for triple classification"""

    def __init__(self, config_path):
        super().__init__(config_path)

    def _split(self):
        return "valid"


class TestDataset(EvaluationDataset):
    """Test split for triple classification"""

    def __init__(self, config_path):
        super().__init__(config_path)

    def _split(self):
        return "test"

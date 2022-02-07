import os
import json
import torch
import faiss

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict

from job.job import Job
from dataset.knowledge_base import KnowledgeBase
from dataset.tokenize import BasePhraseTokenizer
from model.sequence_embedding import BaseSequenceEmbeddingModel
from utils import load_checkpoint, save_checkpoint


class TokenizedEntityPhrases(KnowledgeBase):
    """Tokenize all entity phrases in a knowledge base for k-NN search"""

    def __init__(self, config_path):
        super().__init__(
            config_path, job_key="negater.index", model_key="negater.index.model"
        )

        self.tokenizer = BasePhraseTokenizer.create_for(self)
        self.pretokenize_phrases = self.get_option("dataset.pretokenize_phrases")
        if self.pretokenize_phrases:
            self._pretokenize()

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, i):
        if self.pretokenize_phrases:
            return self.tokenized_entities[i]
        return self.tokenizer.encode(self.entities[i])

    def _load_triples(self):
        pass  # don't need triples for this dataset

    def _pretokenize(self):
        """Pre-tokenize all phrases once and save the tensors to files, or
        load the pretokenized files"""
        phrase_token_path = os.path.join(
            self.dataset_path, self.get_option("dataset.phrases_tokenized_path")
        )
        tokens_filename = os.path.join(phrase_token_path, "phrase_tokens.pt")

        if os.path.exists(tokens_filename):
            self.tokenized_entities = load_checkpoint(
                tokens_filename, device=self.device
            )
            self.log(f"Loaded tokenized entity phrases from {tokens_filename}")
        else:
            self.tokenized_entities = []
            for phrase in tqdm(self.entities, desc="Tokenizing KB phrases"):
                self.tokenized_entities.append(self.tokenizer.encode(phrase))

            if not os.path.isdir(phrase_token_path):
                os.makedirs(phrase_token_path)

            save_checkpoint(self.tokenized_entities, tokens_filename)
            self.log(f"Saved tokenized entity phrases {tokens_filename}")


class NearestNeighborsSearchJob(Job):
    """Build nearest-neighbor index of sequence embeddings"""

    def __init__(self, config_path, job_key="negater.index"):
        super().__init__(
            config_path, job_key="negater.index", model_key="negater.index.model"
        )

        self.k = self.get_option(f"{self.job_key}.k")
        self.batch_size = self.get_option("negater.batch_size")
        self.pool_strategy = self.get_option(f"{self.model_key}.pool_strategy")
        self.save_index = self.get_option(f"{self.job_key}.save")
        self.filename = self.get_option(
            f"{self.job_key}.filename",
            default=os.path.join(self.folder, f"top_{self.k}_index.json"),
        )

        self.dataset = TokenizedEntityPhrases(config_path)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size)

    def _create_model_for_job(self):
        return BaseSequenceEmbeddingModel.create_for(self)

    @torch.no_grad()
    def run(self):
        self.log(f"Building {self.k}-nearest neighbors index...")

        model = self._create_model()

        # Get embeddings for each sequence in the dataset
        hidden_states = torch.zeros(
            (len(self.loader.dataset), model.config.hidden_size),
            dtype=torch.float32,
            device=self.device,
        )

        for i_batch, batch in enumerate(tqdm(self.loader, desc="Encoding text")):
            sequence_embeddings = model(*batch, pool_strategy=self.pool_strategy)[0]
            start = i_batch * self.batch_size
            end = start + len(sequence_embeddings)
            hidden_states[start:end] = sequence_embeddings

        top_k_indices = _top_k_self_similarity_search(
            hidden_states.cpu().numpy(), self.k
        )

        # Save the top-k nearest neighbors to each sequence
        nearest_neighbors = defaultdict(list)

        for sequence, indices in tqdm(
            zip(self.dataset.entities, top_k_indices),
            total=len(top_k_indices),
            desc="Querying k-NN index",
        ):
            nearest_neighbors[sequence] = list(self.dataset.entities[indices])

        if self.save_index:
            with open(self.filename, "w") as f:
                json.dump(nearest_neighbors, f)

                self.log(f"Saved {self.k}-nearest neighbors index to {self.filename}")

        # Log some examples
        examples = list(nearest_neighbors.keys())[:5]

        for key in examples:
            neighbors = nearest_neighbors[key]

            self.log(f"key: '{key}'")
            self.log(f"\t{neighbors}")

        return nearest_neighbors


def _top_k_self_similarity_search(keys, k):
    """Top-k nearest-neighbor search with the same set of keys/values"""
    n, d = keys.shape
    index = faiss.IndexFlatL2(d)
    index.add(keys)
    outputs = index.search(keys, k + 1)  # account for self-similarity
    top_k_indices = outputs[-1]
    return top_k_indices[:, 1:]  # leave out first result for self-similarity

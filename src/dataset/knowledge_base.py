import os

import numpy as np
import pandas as pd

from config import Configurable


class KnowledgeBase(Configurable):
    """KB entities, relations, and triples"""

    def __init__(self, config_path, triples=None, job_key="", model_key=""):
        """Optionally provide triples instead of loading them from files"""
        super().__init__(config_path, job_key=job_key, model_key=model_key)

        self.dataset_path = self.get_option("dataset.path")

        if triples is None:
            self._load_entities_relations()
            self._load_triples()
        else:
            self._set_triples(triples)

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        return self.triples[i]

    def _load_entities_relations(self, extension=".txt"):
        """Create lists of unique entity phrases and relations"""
        data = []
        for fname in ("entity_ids", "relation_ids"):
            lines = []
            with open(os.path.join(self.dataset_path, fname + extension)) as f:
                for line in f:
                    lines.append(line.split("\t")[1].rstrip("\n"))

            data.append(np.asarray(lines))

        self.entities, self.relations = data

    def _load_triples(self, splits=["train", "valid", "test"]):
        """Overload this in child classes based on the split(s) needed"""
        self.triples = np.concatenate(
            [self._load_split(split_name) for split_name in splits]
        )

    def _set_triples(self, triples):
        """Set internal data based on input"""
        self.triples = triples

        entities, relations = set(), set()
        for (head, relation, tail) in self.triples:
            if head not in entities:
                entities.add(head)

            if tail not in entities:
                entities.add(tail)

            if relation not in relations:
                relations.add(relation)

        self.entities, self.relations = list(entities), list(relations)

    def _load_split(self, split_name, extension=".txt", sep="\t"):
        """
        :param split_name: one of 'train', 'valid', 'test'
        :return: numpy array of triples (n_samples, 3)
        """
        fname = os.path.join(self.dataset_path, split_name + extension)
        df = pd.read_csv(
            fname,
            sep=sep,
            names=["head", "relation", "tail"],
        )

        entity_dict = dict(enumerate(self.entities))
        relation_dict = dict(enumerate(self.relations))

        self.log(f"Loaded {split_name} split ({len(df)} examples) from {fname}")

        return pd.concat(
            (
                df["head"].map(entity_dict),
                df["relation"].map(relation_dict),
                df["tail"].map(entity_dict),
            ),
            axis=1,
        ).values

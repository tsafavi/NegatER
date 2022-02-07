import os
import torch
import time

import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from transformers import AdamW

from job.job import Job
from job.generate_negatives import NegativeGenerationJob
from model.triple_classification import BaseTripleClassificationModel


class NegativeRankingJob(Job):
    """Base class for all instantiations of the NegatER framework"""

    def __init__(self, config_path):
        super().__init__(
            config_path, job_key="negater", model_key="negater.fine_tuned_model"
        )

        self.batch_size = self.get_option(f"{self.job_key}.batch_size")
        self.pool_strategy = self.get_option(f"{self.model_key}.pool_strategy")

    def _create_model_for_job(self):
        return BaseTripleClassificationModel.create_for(self)

    def _create_optimizer(self, model, lr, eps):
        optimizer = AdamW(Job.static_optimizer_params(model), lr=lr, eps=eps)
        return optimizer

    def _load_model_and_optimizer(self, output_state_dict=False):
        """Load the fine-tuned model to be mined for negative knowledge"""
        model, state_dict = self._load_model_from_filename(
            self.get_option(f"{self.model_key}.filename"),
            output_state_dict=True,
        )

        model_config = state_dict["config"]
        if "fine_tune" in model_config:
            ft_config = model_config["fine_tune"]
            lr = ft_config["train"]["lr"]
            eps = ft_config["train"]["eps"]
        else:
            lr = self.get_option("fine_tune.train.lr")
            eps = self.get_option("fine_tune.train.eps")

        optimizer = self._create_optimizer(model, lr, eps)
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        outputs = (model, optimizer)

        if output_state_dict:
            outputs += (state_dict,)

        return outputs

    def _save_negatives(self, triples, scores, fname="negatives.tsv"):
        """
        :param triples: list of lists of corrupted (generated) triples
        :param scores: associated scores of triples
        :param fname: filename to save to
        """
        triples = pd.DataFrame.from_records(
            triples, columns=["head", "relation", "tail"]
        )

        triples["score"] = scores
        triples = triples.drop_duplicates(
            subset=["head", "relation", "tail"]
        ).sort_values(
            "score", ascending=False
        )  # ranking step

        fname = os.path.join(self.folder, fname + ".tsv")
        triples.to_csv(fname, sep="\t", index=False)

        self.log(f"Saved {len(triples)} rows to {fname}")


class ThresholdsRankingJob(NegativeRankingJob):
    def __init__(self, config_path, datasets):
        super().__init__(config_path)

        self.datasets = datasets

    @torch.no_grad()
    def run(self):
        model, optimizer, state_dict = self._load_model_and_optimizer(
            output_state_dict=True
        )
        threshold = state_dict["valid_threshold"]

        for relation, threshold in threshold.items():
            if relation in self.datasets:
                relation_dataset = self.datasets[relation]
                loader = DataLoader(relation_dataset, batch_size=self.batch_size)

                negatives, scores = [], []
                for i_batch, batch in enumerate(
                    tqdm(loader, desc=f"Scoring candidates of relation <{relation}>")
                ):
                    triple_scores = (
                        model(*batch, pool_strategy=self.pool_strategy)[0]
                        .view(-1)
                        .cpu()
                        .numpy()
                    )
                    idx = triple_scores < threshold.item()

                    start = i_batch * self.batch_size
                    end = start + len(batch[0])

                    neg_batch = relation_dataset.triples[start:end]
                    negatives.extend(list(neg_batch[idx]))
                    scores.extend(list(triple_scores[idx]))

                self._save_negatives(negatives, scores, fname=f"negatives_{relation}")


class FullGradientsRankingJob(NegativeRankingJob):
    def __init__(self, config_path, dataset):
        super().__init__(config_path)

        self.dataset = dataset
        self.batch_size = 1
        self.loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def run(self):
        model, optimizer = self._load_model_and_optimizer()

        scores = []

        for i_batch, batch in enumerate(tqdm(self.loader, desc="Computing gradients")):
            optimizer.zero_grad()

            triple_scores = model(*batch, pool_strategy=self.pool_strategy)[0].view(-1)
            loss = self.loss(triple_scores, torch.ones_like(triple_scores))
            loss.backward()

            magnitude = 0.0
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    magnitude += torch.norm(parameter.grad).item()
            scores.append(magnitude)

        self._save_negatives(self.dataset.triples, scores)


class ProxyGradientsRankingJob(NegativeRankingJob):
    def __init__(self, config_path, dataset):
        super().__init__(config_path)

        self.job_key = f"{self.job_key}.proxy"

        self.dataset = dataset
        self.init_steps = self.get_option(f"{self.job_key}.init.steps")
        self.proxy_job = ProxyTrainingJob(config_path)

    def run(self):
        model, optimizer = self._load_model_and_optimizer()

        # Individual forward/backward for small subset of data
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        features = torch.zeros(
            (self.init_steps, model.config.hidden_size),
            dtype=torch.float,
            device=self.device,
        )
        targets = torch.zeros(self.init_steps, dtype=torch.float, device=self.device)

        for i, batch in tqdm(
            zip(range(self.init_steps), loader),
            total=self.init_steps,
            desc="Computing full gradients",
        ):
            optimizer.zero_grad()

            outputs = model(*batch, pool_strategy=self.pool_strategy)[1:]
            candidate_embeddings, loss = outputs
            loss.backward()

            magnitude = torch.zeros(1, dtype=torch.float, device=self.device)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    magnitude += torch.norm(parameter.grad)

            features[i] = candidate_embeddings
            targets[i] = magnitude

        # Train the proxy model on the output of the large model
        proxy_dict = self.proxy_job.run(features.detach(), targets)
        proxy_model = proxy_dict["model"]

        prediction_time = -time.time()

        with torch.no_grad():
            # Forward only on the full and proxy models
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
            scores = []

            for batch in tqdm(loader, desc="Approximating gradients"):
                candidate_embeddings = model(*batch, pool_strategy=self.pool_strategy)[
                    1
                ]
                outputs = proxy_model(candidate_embeddings)
                scores.extend(outputs[0].flatten().cpu().numpy().tolist())

        self._save_negatives(self.dataset.triples, scores, fname="negatives_proxy")

        prediction_time += time.time() + proxy_dict["time"]
        self.log(f"Total training + prediction time: {prediction_time:.3f} seconds")


class ProxyTrainingJob(Job):
    """Train a proxy model on the output of a larger model"""

    class ProxyModel(nn.Module):
        def __init__(self, d1=768, d2=100, dropout=0.1):
            super().__init__()

            self.d1 = d1
            self.d2 = d2

            self.W1 = nn.Linear(self.d1, self.d2)
            self.W2 = nn.Linear(self.d2, 1)

            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)

            self.loss = nn.L1Loss()
            # self.loss = nn.MarginRankingLoss(margin=1.0)

        def forward(self, inputs, targets=None):
            scores = self.W2(self.dropout(self.activation(self.W1(inputs))))
            scores = scores.view(-1)

            outputs = (scores,)

            if targets is not None:
                loss = self.loss(scores, targets)
                outputs = outputs + (loss,)

            return outputs

    def __init__(self, config_path):
        super().__init__(config_path, job_key="negater.proxy")

    def run(self, features, targets):
        # load the model and data
        model = ProxyTrainingJob.ProxyModel(
            d2=self.get_option(f"{self.job_key}.d2"),
            dropout=self.get_option(f"{self.job_key}.dropout"),
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.get_option(f"{self.job_key}.train.lr")
        )
        model.to(self.device)

        idx = np.arange(len(features))
        sampler = BatchSampler(
            RandomSampler(idx), self.get_option("negater.batch_size"), drop_last=False
        )

        # train the model on the input features and targets
        training_time = -time.time()
        lowest_loss = None
        n_epochs = self.get_option(f"{self.job_key}.train.epochs")

        for epoch in tqdm(range(n_epochs), desc="Training proxy model"):
            average_loss = 0.0

            epoch_time = -time.time()
            model.train()

            for idx in sampler:
                optimizer.zero_grad()

                outputs = model(features[idx], targets=targets[idx])
                loss = outputs[-1]

                loss.backward()
                optimizer.step()

                average_loss += loss.item()

            epoch_time += time.time()

            average_loss /= len(features)  # average loss per instance

            if lowest_loss is None or average_loss < lowest_loss:
                lowest_loss = average_loss

        training_time += time.time()
        self.log(f"Lowest loss: {lowest_loss}")

        model.eval()
        return {"model": model, "time": training_time}

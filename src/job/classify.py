import os
import time
import datetime
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import AdamW, get_linear_schedule_with_warmup

from job.job import Job
from dataset.triple_classification import (
    FineTuningDataset,
    ValidationDataset,
    TestDataset,
)
from model.triple_classification import BaseTripleClassificationModel


class FineTuneClassificationJob(Job):
    """Base class for fine-tuning language models for triple classification"""

    def __init__(self, config_path):
        super().__init__(config_path, job_key="fine_tune", model_key="fine_tune.model")

        self.pool_strategy = self.get_option(f"{self.model_key}.pool_strategy")

    def _create_model_for_job(self):
        return BaseTripleClassificationModel.create_for(self)


class FineTuneTrainJob(FineTuneClassificationJob):
    """Train a model for triple classification"""

    def __init__(self, config_path):
        super().__init__(config_path)

        self.job_key = f"{self.job_key}.train"

        self.dataset = FineTuningDataset(self.config_path)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.get_option(f"{self.job_key}.batch_size"),
            shuffle=True,
        )

        self.valid_job = FineTuneValidJob(config_path)
        self.save_all_epochs = self.get_option(f"{self.job_key}.save_all_epochs")

        if self.save_all_epochs:
            self.log("Saving all epochs of training")

        self.best_checkpoint = self.get_option(f"{self.job_key}.best_checkpoint")
        self.log(
            f"Best checkpoint will be saved to checkpoint_{self.best_checkpoint}.pt"
        )

    def run(self):
        n_epochs = self.get_option(f"{self.job_key}.n_epochs")

        if self.has_option(f"{self.job_key}.load_from"):
            load_checkpoint = self.get_option(f"{self.job_key}.load_from")
            model, load_dict = self._load_model_from_checkpoint(
                load_checkpoint, output_state_dict=True
            )

            optimizer, scheduler = self._create_optimizer_and_scheduler(model)
            optimizer.load_state_dict(load_dict["optimizer_state_dict"])
            scheduler.load_state_dict(load_dict["scheduler_state_dict"])

            epoch_start = int(load_checkpoint) + 1
            best_accuracy = load_dict["best_accuracy"]

            self.log(f"Resuming training from epoch {epoch_start + 1}")
        else:
            model = self._create_model()
            optimizer, scheduler = self._create_optimizer_and_scheduler(model)
            epoch_start = 0
            best_accuracy = None

            self.log("Training from beginning")

        training_time = -time.time()

        for epoch in range(epoch_start, n_epochs):
            total_loss = 0.0
            epoch_is_best = False

            epoch_time = -time.time()
            model.train()

            for batch in tqdm(self.loader, desc=f"Epoch {epoch + 1}"):
                optimizer.zero_grad()
                outputs = model(*batch, pool_strategy=self.pool_strategy)
                loss = outputs[-1]

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            epoch_time += time.time()
            self.log(f"Loss: {total_loss:.3f}")
            self.log(f"Epoch time: {epoch_time:.3f} sec")

            valid_accuracy, valid_threshold = self.valid_job.run(model=model)

            if best_accuracy is None or valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                epoch_is_best = True

            save_dict = dict(
                timestamp=datetime.datetime.now().strftime(self.dt_format),
                epoch=epoch,
                loss=total_loss,
                epoch_time=epoch_time,
                config=self.options,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                valid_accuracy=valid_accuracy,
                valid_threshold=valid_threshold,
                best_accuracy=best_accuracy,
            )

            if self.save_all_epochs:
                self._save_checkpoint(save_dict, f"{epoch:04}")

            if epoch_is_best:
                self._save_checkpoint(save_dict, self.best_checkpoint)

        training_time += time.time()
        self.log(f"Training time: {training_time} sec")

        model.eval()

    def _create_optimizer(self, model):
        optimizer = AdamW(
            Job.static_optimizer_params(model),
            lr=self.get_option(f"{self.job_key}.lr"),
            eps=self.get_option(f"{self.job_key}.eps"),
        )
        return optimizer

    def _create_optimizer_and_scheduler(self, model):
        optimizer = self._create_optimizer(model)

        num_training_steps = int(
            len(self.dataset) / self.get_option(f"{self.job_key}.batch_size")
        ) * self.get_option(f"{self.job_key}.n_epochs")
        # self.log(f"{num_training_steps} training steps total")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.get_option(f"{self.job_key}.warmup_steps"),
            num_training_steps=num_training_steps,
        )

        return optimizer, scheduler


class FineTuneEvalJob(FineTuneClassificationJob):
    def __init__(self, config_path, job_type):
        super().__init__(config_path)

        self.job_key = f"{self.job_key}.eval"

        job_type = job_type.lower()
        self.job_type = job_type

        self.batch_size = self.get_option(f"{self.job_key}.batch_size")
        self.dataset = (
            ValidationDataset(config_path)
            if self.job_type == "valid"
            else TestDataset(config_path)
        )
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size)

        self.threshold_per_relation = self.get_option(
            f"{self.job_key}.threshold_per_relation"
        )

    @torch.no_grad()
    def run(self):
        raise NotImplementedError

    def _eval_scores(self, model):
        """Get prediction scores for all triples in the dataset in batches"""
        scores = torch.zeros(len(self.dataset), device=self.device, dtype=torch.float32)

        for i_batch, batch in enumerate(
            tqdm(self.loader, desc=f"Evaluating on {self.job_type} split")
        ):
            outputs = model(*batch, pool_strategy=self.pool_strategy)
            logits = outputs[0]

            start = i_batch * self.batch_size
            end = start + len(logits)

            scores[start:end] = logits.view(-1)

        return scores

    def _eval_labels(self):
        return torch.tensor(self.dataset.labels, dtype=torch.long, device=self.device)


class FineTuneValidJob(FineTuneEvalJob):
    """Validate a model for triple classification"""

    def __init__(self, config_path):
        super().__init__(config_path, "valid")

    @torch.no_grad()
    def run(self, model=None):
        """
        :param model: the model to be evaluated. If None, load from the specified
            evaluation file
        :return accuracy: validation accuracy
        :return threshold: float if not self.threshold_per_relation,
            else dict of {relation: relation threshold}
        """
        if model is None:
            best_checkpoint = self.get_option(f"{self.job_key}.test_checkpoint")
            model = self._load_model_from_checkpoint(best_checkpoint)

        model.eval()

        X_valid = self._eval_scores(model)
        y_valid = self._eval_labels()

        # get a global threshold and global accuracy first
        valid_threshold, valid_accuracy = _get_threshold(X_valid, y_valid)

        if self.threshold_per_relation:
            relations = self.dataset.relations
            thresholds = {r: -float("inf") for r in relations}

            y_pred = torch.zeros(y_valid.size(), dtype=torch.long, device=self.device)

            for rel in relations:
                current_rel = self.dataset.triples[:, 1] == rel

                if not np.any(current_rel):
                    thresholds[rel] = valid_threshold
                else:
                    rel_threshold, rel_accuracy = _get_threshold(
                        X_valid[current_rel], y_valid[current_rel]
                    )

                    thresholds[rel] = rel_threshold
                    y_pred[current_rel] = (
                        (X_valid[current_rel] >= rel_threshold).view(-1).long()
                    )

            valid_accuracy = torch.sum(y_valid == y_pred).item() / len(y_pred)
            valid_threshold = thresholds

            self.log(f"\taccuracy with per-relation thresholds: {valid_accuracy}")
        else:
            self.log(
                f"\tglobal validation threshold: {valid_threshold}"
                f" | global validation accuracy: {valid_accuracy}"
            )

        return valid_accuracy, valid_threshold


class FineTuneTestJob(FineTuneEvalJob):
    """Test a triple classification model"""

    def __init__(self, config_path):
        super().__init__(config_path, "test")

    @torch.no_grad()
    def run(self):
        """Load the best-performing trained model and evaluate on the test set"""
        best_checkpoint = self.get_option(f"{self.job_key}.test_checkpoint")

        model, state_dict = self._load_model_from_checkpoint(
            best_checkpoint, output_state_dict=True
        )
        valid_threshold = state_dict["valid_threshold"]

        if isinstance(valid_threshold, float):
            self.log(f"\tvalid threshold: {valid_threshold}")

        X_test = self._eval_scores(model)
        y_test = self._eval_labels()
        X_thresh = torch.zeros(X_test.size(), dtype=torch.float, device=self.device)

        if self.threshold_per_relation:
            y_pred_test = torch.zeros(
                y_test.size(), dtype=torch.long, device=self.device
            )
            relations = self.dataset.relations

            for rel in relations:
                current_rel = self.dataset.triples[:, 1] == rel
                rel_threshold = valid_threshold[rel]
                y_pred_test[current_rel] = (
                    (X_test[current_rel] >= rel_threshold).view(-1).long()
                )
                X_thresh[current_rel] = rel_threshold
        else:
            y_pred_test = (X_test >= valid_threshold).long()
            X_thresh *= valid_threshold

        y_test = y_test.cpu().numpy()
        y_pred_test = y_pred_test.cpu().numpy()

        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        accuracy = accuracy_score(y_test, y_pred_test)

        self.log(
            f"\ttest precision: {precision:.4f} | recall: {recall:.4f} | "
            f"accuracy: {accuracy:.4f}"
        )

        df = pd.DataFrame(
            {
                "head": self.dataset.triples[:, 0],
                "relation": self.dataset.triples[:, 1],
                "tail": self.dataset.triples[:, 2],
                "threshold": X_thresh.cpu().numpy(),
                "score": X_test.cpu().numpy(),
                "y_true": y_test,
                "y_pred": y_pred_test,
            }
        )

        if self.get_option(f"{self.job_key}.save_predictions"):
            df.to_csv(
                os.path.join(self.folder, "predictions.tsv"), sep="\t", index=False
            )


def _get_threshold(scores, labels):
    """
    :param scores: torch.tensor of prediction scores
    :param labels: torch.tensor of triple labels
    :return threshold: best decision threshold for these scores
    """
    best_accuracy = best_threshold = None

    for threshold in scores:
        y_pred = scores >= threshold
        accuracy = torch.sum(y_pred == labels).item() / len(labels)

        if best_accuracy is None or accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

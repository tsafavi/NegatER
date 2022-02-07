import os

from transformers import AdamW

from config import Configurable
from utils import load_checkpoint, save_checkpoint


class Job(Configurable):
    """Base class for all jobs"""

    def __init__(self, config_path, job_key="", model_key=""):
        super().__init__(config_path, job_key=job_key, model_key=model_key)

    def run(self):
        raise NotImplementedError

    def _create_model_for_job(self):
        # Override this to return the type of LM used by this job, if any
        raise NotImplementedError

    def _create_model(self):
        model = self._create_model_for_job()
        model.to(self.device)
        model.eval()
        return model

    def _save_checkpoint(self, save_dict, checkpoint_name, log=True):
        checkpoint_file = os.path.join(self.folder, f"checkpoint_{checkpoint_name}.pt")
        save_checkpoint(save_dict, checkpoint_file)

        if log:
            self.log(f"Saved checkpoint to {checkpoint_file}")

    def _load_checkpoint(self, filename, log=True):
        checkpoint = load_checkpoint(filename, device=self.device)

        if log:
            self.log(f"Loaded checkpoint from {filename}")
        return checkpoint

    def _load_model_from_checkpoint(self, checkpoint_name, output_state_dict=False):
        filename = os.path.join(self.folder, f"checkpoint_{checkpoint_name}.pt")
        return self._load_model_from_filename(
            filename, output_state_dict=output_state_dict
        )

    def _load_model_from_filename(self, filename, output_state_dict=False):
        state_dict = self._load_checkpoint(filename)

        model = self._create_model()
        model.load_state_dict(state_dict["model_state_dict"])

        if output_state_dict:
            outputs = (model, state_dict)
            return outputs

        return model

    @staticmethod
    def static_optimizer_params(model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

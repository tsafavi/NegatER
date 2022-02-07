import torch.nn as nn

from transformers import (
    BertPreTrainedModel,
    BertModel,
    RobertaModel,
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from model.pool import pool_lm_output


class BaseTripleClassificationModel(object):
    def __init__(self, config):
        super().__init__(config)

        self.triple_scoring_head = nn.Linear(config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def create_for(job):
        model_key = job.model_key
        pretrained_name = job.get_option(f"{model_key}.pretrained_name")

        if "roberta" in pretrained_name:
            return RobertaForTripleClassification.from_pretrained(pretrained_name)
        elif "bert" in pretrained_name:
            return BertForTripleClassification.from_pretrained(pretrained_name)

        raise ValueError(f"pretrained_name={pretrained_name} not recognized")

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        triple_labels=None,
        pool_strategy="cls",
    ):
        """
        :param input_ids: torch.tensor of torch.long token input IDs (n, max_seq_len)
        :param attention_mask: torch.tensor of torch.long attention mask
            (n, max_seq_len)
        :param token_type_ids: torch.tensor of torch.long segment IDs (n, max_seq_len)
        :param triple_labels: optional torch.tensor of torch.long triple labels (n,)
        :param pool_strategy: how to pool hidden states to get triple representations
        :return triple_scores: scores of input triples
        :return triple_embeddings: embeddings of triples
        :return loss: if triple_labels is not None, BCE loss between labels/logits
        """
        model_output = self._base_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
        )

        triple_embeddings = pool_lm_output(pool_strategy, model_output, attention_mask)
        triple_scores = self.triple_scoring_head(triple_embeddings)
        outputs = (
            triple_scores,
            triple_embeddings,
        )

        if triple_labels is not None:
            loss = self.loss_fn(triple_scores.view(-1), triple_labels.float().view(-1))
            outputs = outputs + (loss,)

        return outputs

    def _base_model(self):
        raise NotImplementedError


class BertForTripleClassification(BaseTripleClassificationModel, BertPreTrainedModel):
    """Fine-tune BERT to score knowledge base triples"""

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def _base_model(self):
        return self.bert


class RobertaForTripleClassification(
    BaseTripleClassificationModel, RobertaPreTrainedModel
):
    """Fine-tune RoBERTa to score knowledge base triples"""

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def _base_model(self):
        return self.roberta

from transformers import BertPreTrainedModel, BertModel, RobertaModel

from model.pool import pool_lm_output


class BaseSequenceEmbeddingModel(BertPreTrainedModel):
    """Run phrases through encoder and take [CLS] embeddings as representations"""

    def __init__(self, config):
        super().__init__(config)

        self._init_model(config)
        self.init_weights()

    @staticmethod
    def create_for(job):
        """Static factory method"""
        pretrained_name = job.get_option(f"{job.model_key}.pretrained_name")

        if "roberta" in pretrained_name:
            return RobertaForSequenceEmbedding.from_pretrained(pretrained_name)
        elif "bert" in pretrained_name:
            return BertForSequenceEmbedding.from_pretrained(pretrained_name)

        raise ValueError(f"pretrained_name={pretrained_name} not recognized")

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pool_strategy="cls",
    ):
        model_output = self._get_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
        )

        sequence_embeddings = pool_lm_output(
            pool_strategy, model_output, attention_mask
        )
        outputs = (sequence_embeddings,)

        return outputs

    def _init_model(self):
        raise NotImplementedError


class BertForSequenceEmbedding(BaseSequenceEmbeddingModel):
    """Uses a BERT encoder"""

    def __init__(self, config):
        super().__init__(config)

    def _init_model(self, config):
        self.bert = BertModel(config)

    def _get_model(self):
        return self.bert


class RobertaForSequenceEmbedding(BaseSequenceEmbeddingModel):
    """Uses a RoBERTa encoder"""

    def __init__(self, config):
        super().__init__(config)

    def _init_model(self, config):
        self.roberta = RobertaModel(config)

    def _get_model(self):
        return self.roberta

import torch

from transformers import BertTokenizer, RobertaTokenizer

from config import Configurable


class BaseTokenizer(Configurable):
    """Base class for LM tokenization of text sequences"""

    def __init__(self, dataset):
        """The tokenizer constructors shouldn't be called directly.
        Rather, use the static create_for method"""
        super().__init__(dataset.config_path)

        self.base_ = self._base_tokenizer().from_pretrained(
            dataset.get_option(f"{dataset.model_key}.pretrained_name")
        )
        self.max_seq_len = dataset.get_option(f"{dataset.model_key}.max_seq_len")

        self.cls_token = self.base_.cls_token
        self.sep_token = self.base_.sep_token
        self.mask_token = self.base_.mask_token

    @staticmethod
    def create_for(dataset):
        """Static factory method"""
        raise NotImplementedError

    def encode(self, inputs):
        """Encode inputs into IDs following LM's convention"""
        raise NotImplementedError

    def _base_tokenizer(self):
        """Should return the base tokenizer type"""
        raise NotImplementedError

    def _encode_tokenized_text(self, tokens_list):
        """
        :param tokens_list: list of list of str
        :return input_ids: torch.tensor of IDs for all tokens in the triple
        :return attention_mask: torch.tensor of mask of 1's for token IDs
        :return token_type_ids: torch.tensor of all 0's
        """
        flat_tokens = [self.cls_token]
        for tokens in tokens_list:
            flat_tokens += tokens
            flat_tokens += [self.sep_token]

        npad = self.max_seq_len - len(flat_tokens)

        # construct core inputs and pad to max length
        input_ids = self.base_.convert_tokens_to_ids(flat_tokens) + [0] * npad
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)

        attention_mask = [1] * len(flat_tokens) + [0] * npad
        attention_mask = torch.tensor(
            attention_mask, dtype=torch.long, device=self.device
        )

        token_type_ids = torch.zeros(
            input_ids.size(), dtype=torch.long, device=self.device
        )

        outputs = (input_ids, attention_mask, token_type_ids)
        return outputs


class BaseTripleTokenizer(BaseTokenizer):
    """Class for tokenization of commonsense (head, relation, tail) triples"""

    def __init__(self, dataset):
        super().__init__(dataset)

    @staticmethod
    def create_for(dataset):
        pretrained_name = dataset.get_option(f"{dataset.model_key}.pretrained_name")

        if "roberta" in pretrained_name:
            return RobertaTripleTokenizer(dataset)
        elif "bert" in pretrained_name:
            return BertTripleTokenizer(dataset)

        raise ValueError(f"pretrained_name={pretrained_name} not recognized")

    def encode(self, inputs, triple_label=None, is_pretokenized=False):
        """
        :param inputs: (head, relation, tail) triple or tuple of tokens
        :param triple_label: optional 0/1 label of triple
        :param is_pretokenized: whether the input is already tokenized
        :return input_ids: IDs of all tokens in the triple
        :return attention_mask: mask of 1's for tokens and 0's for padding
        :return token_type_ids: segment IDs (all 0)
        :return triple_label: if triple_label is not None, label of triple
        """
        if not is_pretokenized:
            tokenized = [self.base_.tokenize(tokens) for tokens in inputs]

            # truncate triple to appropriate length
            head, relation, tail = [tokenized[i] for i in range(3)]
            while True:
                total_length = len(head) + len(relation) + len(tail)
                if total_length <= self.max_seq_len - 4:
                    break
                if len(head) > len(tail):
                    head.pop()
                else:
                    tail.pop()

            outputs = self._encode_tokenized_text(tokenized)
        else:
            outputs = inputs

        # Add the label of the sample to the encoding
        if triple_label is not None:
            outputs = outputs + (
                torch.tensor(triple_label, dtype=torch.long, device=self.device),
            )

        return outputs


class BertTripleTokenizer(BaseTripleTokenizer):
    """Class for BERT tokenization of triples"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def _base_tokenizer(self):
        return BertTokenizer


class RobertaTripleTokenizer(BaseTripleTokenizer):
    """Class for RoBERTa tokenization of triples"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def _base_tokenizer(self):
        return RobertaTokenizer


class BasePhraseTokenizer(BaseTokenizer):
    """Class for tokenization of text strings"""

    def __init__(self, dataset):
        super().__init__(dataset)

    @staticmethod
    def create_for(dataset):
        pretrained_name = dataset.get_option(f"{dataset.model_key}.pretrained_name")

        if "roberta" in pretrained_name:
            return RobertaPhraseTokenizer(dataset)
        elif "bert" in pretrained_name:
            return BertPhraseTokenizer(dataset)

        raise ValueError(f"pretrained_name={pretrained_name} not recognized")

    def encode(self, inputs, is_pretokenized=False):
        """
        :param inputs: string of text or tuple of tokens
        :param is_pretokenized: whether the input is already tokenized
        :return input_ids: IDs of all tokens in the triple
        :return attention_mask: mask of 1's for tokens and 0's for padding
        :return token_type_ids: segment IDs (all 0)
        """
        if not is_pretokenized:
            # convert to [CLS] text [SEP]
            tokenized = self.base_.tokenize(inputs)
            while len(tokenized) > self.max_seq_len - 2:
                tokenized.pop()

            outputs = self._encode_tokenized_text([tokenized])
            return outputs
        
        return inputs


class BertPhraseTokenizer(BasePhraseTokenizer):
    """Class for BERT tokenization of text strings"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def _base_tokenizer(self):
        return BertTokenizer


class RobertaPhraseTokenizer(BasePhraseTokenizer):
    """Class for RoBERTa tokenization of text strings"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def _base_tokenizer(self):
        return RobertaTokenizer

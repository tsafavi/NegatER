import torch


def mean_pooling(model_output, attention_mask):
    """Credit: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens"""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(model_output, attention_mask):
    """Credit: https://huggingface.co/sentence-transformers/bert-base-nli-max-tokens"""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    token_embeddings[
        input_mask_expanded == 0
    ] = -1e9  # Set padding tokens to small negative value
    max_over_time = torch.max(token_embeddings, 1)[0]
    return max_over_time


def cls_pooling(model_output):
    """Return the [CLS] hidden states of each input"""
    return model_output[0][:, 0]


def pool_lm_output(pool_strategy, model_output, attention_mask):
    if pool_strategy == "mean":
        return mean_pooling(model_output, attention_mask)
    elif pool_strategy == "max":
        return max_pooling(model_output, attention_mask)
    elif pool_strategy == "cls":
        return cls_pooling(model_output)

    raise ValueError(f"pool_strategy={pool_strategy} not supported")

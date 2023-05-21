import numpy as np
import torch


def embed_bert_cls(text, model, tokenizer, max_length=128):
    t = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}

    with torch.no_grad():
        model_output = model(**t)

    # embeddings = model_output.pooler_output  # do not use pooler because it has one unused layer
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return {'cls': embeddings[0].cpu().numpy()}


def embed_bert_cls2(text, model, tokenizer, max_length=128):
    t = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}
    with torch.no_grad():
        model_output = model(**t)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def mean_pooling(model_output, attention_mask, norm=False):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sums = sum_embeddings / sum_mask
    if norm:
        sums = torch.nn.functional.normalize(sums)
    return sums


def embed_bert_pool(text, model, tokenizer, max_length=128, norm=False):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    if norm:
        embeddings = torch.nn.functional.normalize(embeddings)
    return {'mean': embeddings[0].cpu().numpy()}


def embed_bert_both(text, model, tokenizer, max_length=128):
    t = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}
    with torch.no_grad():
        model_output = model(**t)
    cls_emb = model_output.last_hidden_state[:, 0, :]
    cls_emb_norm = torch.nn.functional.normalize(cls_emb)
    mean_emb = mean_pooling(model_output, t['attention_mask'])
    mean_emb_norm = torch.nn.functional.normalize(mean_emb)
    result = {
        'cls': cls_emb[0].cpu().numpy(),
        'mean': mean_emb[0].cpu().numpy(),
        'cls_norm': cls_emb_norm[0].cpu().numpy(),
        'mean_norm': mean_emb_norm[0].cpu().numpy(),
    }
    if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
        po = model_output.pooler_output
        po_normed = torch.nn.functional.normalize(po)
        result['pooler'] = po[0].cpu().numpy()
        result['pooler_norm'] = po_normed[0].cpu().numpy()
    return result


def get_word_vectors_with_bert(words, model, tokenizer, return_raw=False):
    """
    Take list of words (or other tokens) as an input.
    Return either a matrix of token embeddings and its corresponding word ids,
    or a dict from word id to its average vector.
    Can be used to evaluate feature extractors for NER and other sequence labeling problems.
    """
    b = tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=True).to(model.device)
    with torch.no_grad():
        model_output = model(**b)
    vectors = model_output.last_hidden_state[0, :, :].cpu().numpy()
    word_ids = b.word_ids()
    if return_raw:
        return vectors, word_ids

    id2vecs = {i: [] for i, _ in enumerate(words)}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            id2vecs[word_id].append(vectors[i])
    for i in sorted(id2vecs.keys()):
        if len(id2vecs[i]) == 0:
            id2vecs[i] = np.zeros(model.config.hidden_size)
        else:
            id2vecs[i] = np.mean(id2vecs[i], 0)
    return id2vecs

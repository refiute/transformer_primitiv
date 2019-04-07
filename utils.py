# coding: utf-8

import random
import numpy as np
from primitiv import functions as F

def load_corpus(filepath, tokenizer):
    with open(filepath) as ifs:
        corpus = [[tokenizer.PieceToId(piece) for piece in line.strip().split(" ")]
                  for line in ifs]
    return corpus

def clean_corpus(src, trg, config):
    clean_src = []
    clean_trg = []
    max_len = config['generation_limit']
    ratio = config['ratio']
    for src_sent, trg_sent in zip(src, trg):
        src_len = len(src_sent)
        trg_len = len(trg_sent)
        if src_len > max_len or trg_len > max_len or \
                src_len / trg_len > ratio or trg_len / src_len > ratio:
            continue
        clean_src.append(src_sent)
        clean_trg.append(trg_sent)
    return clean_src, clean_trg

def create_batch_itr(src, max_tokens=1e9, max_sentences=1e9, shuffle=False):
    indice = list(range(len(src)))
    if shuffle:
        random.shuffle(indice)

    batch = []
    max_len = 0
    for idx in indice:
        max_len = max(max_len, len(src[idx]))
        assert max_len <= max_tokens, "sentence at index {} exceeds max_tokens limit!".format(idx)
        num_tokens = (len(batch) + 1) * max_len

        if num_tokens > max_tokens or len(batch) == max_sentences:
            yield batch
            batch = [idx]
            max_len = 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch

def make_batch(corpus, sent_ids, eos_id):
    batch_size = len(sent_ids)
    max_len = 0
    for sid in sent_ids:
        max_len = max(max_len, len(corpus[sid]))
    batch = [[eos_id] * batch_size for i in range(max_len)]
    for i in range(batch_size):
        sent = corpus[sent_ids[i]]
        for j in range(len(sent)):
            batch[j][i] = sent[j]
    return batch

def subsequent_mask(size):
    return np.triu(np.ones((size, size)), k=1) == 1

def padding_mask(batch, eos_id):
    seq_len = len(batch)
    batch_size = len(batch[0])

    mask = [np.array([[False] * seq_len]) for _ in range(batch_size)]
    for j in range(batch_size):
        pad_flag = False
        for i in range(seq_len):
            if batch[i][j] == eos_id:
                mask[j][0][i] = pad_flag
                pad_flag = True

    return mask

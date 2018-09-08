# coding: utf-8

import numpy as np
from primitiv import functions as F

def load_corpus(filepath, tokenizer):
    with open(filepath) as ifs:
        corpus = [[tokenizer.PieceToId(piece) for piece in line.strip().split(" ")]
                  for line in ifs]
    return corpus

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
    for i in range(seq_len):
        for j in range(batch_size):
            mask[j][0][i] = (batch[i][j] == eos_id)

    return mask

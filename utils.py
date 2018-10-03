# coding: utf-8

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

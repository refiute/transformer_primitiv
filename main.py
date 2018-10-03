#!/usr/bin/env python
# coding: utf-8

import sys
import random
import math
import json

from argparse import ArgumentParser
from configparser import ConfigParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from primitiv import Device, Graph, Optimizer, Shape
from primitiv import devices as D
from primitiv import optimizers as O
from primitiv import tensor_functions as TF

import sentencepiece as spm

from model import Transformer
from preproc import preproc
from utils import load_corpus, clean_corpus, make_batch, subsequent_mask, padding_mask

def train(model, optimizer, config, best_valid):
    max_epoch = config["epoch"]
    batchsize = config["batchsize"]

    optimizer.add(model)

    corpus_prefix = Path(config['corpus_prefix']) / "subword"
    model_path = corpus_prefix / "spm.model"
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path))
    train_src = load_corpus(corpus_prefix / Path(config["train_source"]).name, tokenizer)
    train_trg = load_corpus(corpus_prefix / Path(config["train_target"]).name, tokenizer)
    train_src, train_trg = clean_corpus(train_src, train_trg, config)
    dev_src = load_corpus(corpus_prefix / Path(config["dev_source"]).name, tokenizer)
    dev_trg = load_corpus(corpus_prefix / Path(config["dev_target"]).name, tokenizer)
    dev_src, dev_trg = clean_corpus(dev_src, dev_trg, config)
    num_train_sents = len(train_src)
    num_dev_sents = len(dev_src)

    eos_id = tokenizer.eos_id()
    train_ids = list(range(num_train_sents))
    dev_ids = list(range(num_dev_sents))

    for epoch in range(max_epoch):
        g = Graph()
        Graph.set_default(g)

        print("epoch: %2d/%2d" % (epoch + 1, max_epoch))
        print("\tlearning rate scale = %.4e" % (optimizer.get_learning_rate_scaling()))

        random.shuffle(train_ids)

        train_loss = 0.
        train_itr = tqdm(range(0, num_train_sents, batchsize), desc='train')
        for ofs in train_itr:
            step_num = optimizer.get_epoch() + 1
            new_scale = 1 / math.sqrt(config['d_model']) * \
                min(1 / math.sqrt(step_num),
                    step_num * math.pow(config['warmup_steps'], -1.5))
            optimizer.set_learning_rate_scaling(new_scale)

            batch_ids = train_ids[ofs : min(num_train_sents, ofs + batchsize)]
            src_batch = make_batch(train_src, batch_ids, eos_id)
            trg_batch = make_batch(train_trg, batch_ids, eos_id)
            src_mask = padding_mask(src_batch, eos_id)
            trg_mask = [x | subsequent_mask(len(trg_batch) - 1) for x in padding_mask(trg_batch[:-1], eos_id)]

            g.clear()
            loss = model.loss(src_batch, trg_batch, src_mask, trg_mask)
            train_loss += loss.to_float() * len(batch_ids)
            train_itr.set_postfix(loss=loss.to_float())

            optimizer.reset_gradients()
            loss.backward()
            optimizer.update()
        print("\ttrain loss = %.4f" % (train_loss / num_train_sents))

        g.clear()
        valid_loss = 0.
        valid_itr = tqdm(range(0, num_dev_sents, batchsize), desc='valid')
        for ofs in valid_itr:
            batch_ids = dev_ids[ofs : min(ofs + batchsize, num_dev_sents)]
            src_batch = make_batch(dev_src, batch_ids, eos_id)
            trg_batch = make_batch(dev_trg, batch_ids, eos_id)
            src_mask = padding_mask(src_batch, eos_id)
            trg_mask = [x | subsequent_mask(len(trg_batch) - 1) for x in padding_mask(trg_batch[:-1], eos_id)]

            loss = model.loss(src_batch, trg_batch, src_mask, trg_mask, train=False)
            valid_loss += loss.to_float() * len(batch_ids)
            valid_itr.set_postfix(loss=loss.to_float())
        print("\tvalid loss = %.4f" % (valid_loss / num_dev_sents))

        if valid_loss < best_valid:
            best_valid = valid_loss
            print('\tsaving model/optimizer ... ', end="", flush=True)
            prefix = config['model_prefix']
            model.save(prefix + '.model')
            optimizer.save(prefix + '.optimizer')
            with Path(prefix).with_suffix('.valid').open('w') as f:
                f.write(str(best_valid))
            print('done.')

def test(model, config):
    batchsize = config["batchsize"]

    corpus_prefix = Path(config['corpus_prefix']) / "subword"
    model_path = corpus_prefix / "spm.model"
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path))
    test_src = load_corpus(corpus_prefix / Path(config["test_source"]).name, tokenizer)
    num_test_sents = len(test_src)

    eos_id = tokenizer.eos_id()
    test_ids = list(range(num_test_sents))

    test_itr = tqdm(range(0, num_test_sents, batchsize), desc='test')
    for ofs in test_itr:
        batch_ids = test_ids[ofs : min(ofs + batchsize, num_test_sents)]
        src_batch = make_batch(test_src, batch_ids, eos_id)
        src_mask = padding_mask(src_batch, eos_id)
        src_encode = model.encode(src_batch, src_mask, train=False)

        trg_ids = [np.array([tokenizer.PieceToId('<s>')] * len(batch_ids))]
        eos_ids = np.array([eos_id] * len(batch_ids))
        while (trg_ids[-1] != eos_ids).any():
            if len(trg_ids) > config['generation_limit']:
                print("Warning: Sentence generation did not finish in", config['generation_limit'],
                      "iterations.", file=sys.stderr)
                trg_ids.append(eos_ids)
                break

            trg_mask = [subsequent_mask(len(trg_ids)) for _ in padding_mask(trg_ids, eos_id)]
            out = model.decode(src_encode, trg_ids, src_mask, trg_mask, train=False)
            y = TF.pick(out, [out.shape()[0] - 1], 0)
            y = np.array(y.argmax(1))
            trg_ids.append(y)

        hyp = [hyp_sent[:np.where(hyp_sent == eos_id)[0][0]] for hyp_sent in np.array(trg_ids).T]
        for ids in hyp:
            sent = tokenizer.DecodeIds(ids.tolist())
            print(sent)


def main(config):
    mode = config['mode']
    if mode == 'preproc':
        preproc(config)
        return

    print('initializing device ...', end='', file=sys.stderr, flush=True)
    dev = D.Naive() if config['gpu'] < 0 else D.CUDA(config['gpu'])
    Device.set_default(dev)
    print("done.", file=sys.stderr, flush=True)

    prefix = config['model_prefix']
    if mode == 'train':
        model = Transformer(config['n_heads'], config['n_stacks'], config['dropout'], config['max_len'])
        model.init(config['vocabulary_size'], config['d_model'], config['d_ff'])
        optimizer = O.Adam(alpha=1, beta2=0.98, eps=1e-9)
        optimizer.set_gradient_clipping(5)
        train(model, optimizer, config, 1e10)
    elif mode == 'resume':
        print('loading model/optimizer ... ', end='', file=sys.stderr, flush=True)
        model = Transformer(config['n_heads'], config['n_stacks'], config['dropout'], config['max_len'])
        model.load(prefix + '.model')
        optimizer = O.Adam(alpha=1, beta2=0.98, eps=1e-9)
        optimizer.load(prefix + '.optimizer')
        with Path(prefix).with_suffix('.valid').open() as f:
            valid_ppl = float(f.read().strip())
        print('done.', file=sys.stderr, flush=True)
        train(model, optimizer, config, valid_ppl)
    elif mode == 'test':
        model = Transformer(config['n_heads'], config['n_stacks'], config['dropout'], config['max_len'])
        model.load(prefix + '.model')
        test(model, config)

def get_config():
    parser = ArgumentParser()
    parser.add_argument('mode', help="'train', 'resume', 'test', or 'preproc'")
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()

    config = json.load(open(args.config))
    config['mode'] = args.mode
    return config

if __name__ == '__main__':
    config = get_config()
    main(config)

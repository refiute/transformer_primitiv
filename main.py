#!/usr/bin/env python
# coding: utf-8

import sys
import random
import math
import json

from argparse import ArgumentParser
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
from utils import (
    load_corpus,
    clean_corpus,
    make_batch,
    subsequent_mask,
    padding_mask,
    create_batch_itr
)

def train(model, optimizer, config, best_valid):
    max_epoch = config.get("max_epoch", int(1e9))
    max_iteration = config.get("max_iteration", int(1e9))
    max_sentences = config.get("max_sentences", 1e9)
    max_tokens = config.get("max_tokens", 1e9)
    update_freq = config.get('update_freq', 1)

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

    epoch = 0
    iteration = 0
    while epoch < max_epoch and iteration < max_iteration:
        g = Graph()
        Graph.set_default(g)

        train_itr = create_batch_itr(train_src, train_trg, max_tokens, max_sentences, shuffle=True)
        train_itr = tqdm(train_itr, desc='train epoch {}'.format(epoch + 1))

        train_loss = 0.
        itr_loss = 0.
        itr_tokens = 0
        itr_sentences = 0
        optimizer.reset_gradients()
        for step, batch_ids in enumerate(train_itr):
            src_batch = make_batch(train_src, batch_ids, eos_id)
            trg_batch = make_batch(train_trg, batch_ids, eos_id)
            src_mask = padding_mask(src_batch, eos_id)
            trg_mask = [x | subsequent_mask(len(trg_batch) - 1) for x in padding_mask(trg_batch[:-1], eos_id)]
            itr_tokens += len(src_batch) * len(src_batch[0])
            itr_sentences += len(batch_ids)

            g.clear()
            loss = model.loss(src_batch, trg_batch, src_mask, trg_mask)

            loss /= update_freq
            loss.backward()
            loss_val = loss.to_float()
            train_loss += loss_val * update_freq * len(batch_ids)
            itr_loss += loss_val

            # with open('graph.dot', 'w') as f:
            #     print(g.dump("dot"), end="", file=f)

            if (step + 1) % update_freq == 0:
                step_num = optimizer.get_epoch() + 1
                new_scale = config['d_model'] ** (-0.5) * \
                    min(step_num ** (-0.5), step_num * config['warmup_steps'] ** (-1.5))
                optimizer.set_learning_rate_scaling(new_scale)

                optimizer.update()
                optimizer.reset_gradients()

                iteration += 1
                train_itr.set_postfix(
                    itr=("%d" % (iteration)),
                    loss=("%.3lf" % (itr_loss)),
                    wpb=("%d" % (itr_tokens)),
                    spb=("%d" % (itr_sentences)),
                    lr=optimizer.get_learning_rate_scaling()
                )
                itr_loss = 0.
                itr_tokens = 0
                itr_sentences = 0

            if iteration >= max_iteration:
                break
        print("\ttrain loss = %.4f" % (train_loss / num_train_sents))


        g.clear()
        valid_loss = 0.
        valid_itr = create_batch_itr(dev_src, dev_trg, max_tokens, max_sentences, shuffle=False)
        valid_itr = tqdm(valid_itr, desc='valid epoch {}'.format(epoch + 1))
        for batch_ids in valid_itr:
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
    max_sentences = config.get("max_sentences", 1e9)
    max_tokens = config.get("max_tokens", 1e9)

    corpus_prefix = Path(config['corpus_prefix']) / "subword"
    model_path = corpus_prefix / "spm.model"
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path))
    test_src = load_corpus(corpus_prefix / Path(config["test_source"]).name, tokenizer)
    num_test_sents = len(test_src)

    eos_id = tokenizer.eos_id()
    test_ids = list(range(num_test_sents))

    test_itr = create_batch_itr(test_src, max_tokens=max_tokens,
                                max_sentences=max_sentences, shuffle=False)
    test_itr = tqdm(test_itr, desc='test')
    for batch_ids in test_itr:
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
        model = Transformer(config['n_heads'], config['n_stacks'], config['dropout'], config['generation_limit'])
        model.init(config['vocabulary_size'], config['d_model'], config['d_ff'])
        optimizer = O.Adam(alpha=1, beta2=0.98, eps=1e-9)
        optimizer.set_gradient_clipping(5)
        train(model, optimizer, config, 1e10)
    elif mode == 'resume':
        print('loading model/optimizer ... ', end='', file=sys.stderr, flush=True)
        model = Transformer(config['n_heads'], config['n_stacks'], config['dropout'], config['generation_limit'])
        model.load(prefix + '.model')
        optimizer = O.Adam(alpha=1, beta2=0.98, eps=1e-9)
        optimizer.set_gradient_clipping(5)
        optimizer.load(prefix + '.optimizer')
        with Path(prefix).with_suffix('.valid').open() as f:
            valid_ppl = float(f.read().strip())
        print('done.', file=sys.stderr, flush=True)
        train(model, optimizer, config, valid_ppl)
    elif mode == 'test':
        model = Transformer(config['n_heads'], config['n_stacks'], config['dropout'], config['generation_limit'])
        model.load(prefix + '.model')
        test(model, config)

def get_config():
    parser = ArgumentParser()
    parser.add_argument('mode', help="'train', 'resume', 'test', or 'preproc'")
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()

    config = json.load(open(args.config))
    config['mode'] = args.mode

    for k, v in config.items():
        print("{}: {}".format(k, v), file=sys.stderr)

    return config

if __name__ == '__main__':
    config = get_config()
    main(config)

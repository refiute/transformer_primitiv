#!/usr/bin/env python
# coding: utf-8

import sys
import random
import math
import json

from argparse import ArgumentParser
from configparser import ConfigParser
from collections import defaultdict

import numpy as np

from primitiv import Device, Graph, Optimizer
from primitiv import devices as D
from primitiv import optimizers as O

from preproc import preproc
from model import Transformer

def train(model, optimizer, config):
    pass

def main(config):
    mode = config['mode']
    if mode == 'preproc':
        preproc(config)
        return

    print('initializing device ...', end='', file=sys.stderr, flush=True)
    dev = D.Naive() if config['global']['gpu'] < 0 else D.CUDA(config['global']['gpu'])
    Device.set_default(dev)
    print("done.", file=sys.stderr, flush=True)

    prefix = config.model
    if mode == 'train':
        model = Transformer()
        model.init()
        optimizer = O.Adam(beta2=0.98, eps=1e-9)
        train(model, optimizer, config, 1e10)
    elif mode == 'resume':
        print('loading model/optimizer ... ', end='', file=sys.stderr, flush=True)
        model = Transformer()
        model.load(prefix + '.model')
        optimizer = O.Adam(beta2=0.98, eps=1e-9)
        optimizer.load(prefix + '.optimizer')
        valid_ppl = load_ppl(prefix + '.valid_ppl')
        print('done.', file=sys.stderr, flush=True)
        train(model, optimizer, valid_ppl)
    elif mode == 'test':
        pass

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

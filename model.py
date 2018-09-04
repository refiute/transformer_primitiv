# coding: utf-8

import math
import numpy as np

from primitiv import Parameter, Model, Shape
from primitiv import functions as F
from primitiv import initializers as I

class LayerNorm(Model):
    def __init__(self, eps=1e-6):
        self.eps = eps

        self.pgain = Parameter()
        self.pbias = Parameter()
        self.scan_attributes()

    def init(self, d_model):
        self.pgain.init([d_model], I.Constant(1))
        self.pbias.init([d_model], I.Constant(0))

    def __call__(self, x):
        gain = F.parameter(self.pgain)
        bias = F.parameter(self.pbias)

        mean = F.mean(x, -1)
        std = F.mean(x * x, -1) - mean * mean

        return gain * (x - mean) / (std + self.eps) + bias

class ScaledDotProductAttention():
    def __init__(self, d_k, dropout):
        self.dropout = dropout

    def __call__(self, d_k, query, key, value, mask, train):
        attn = (query @ F.transpose(key)) / math.sqrt(d_k)

        if mask is not None:
            attn -= -2000 * mask

        attn_prob = F.softmax(attn)
        attn_prob = F.dropout(attn_prob, self.dropout, train)
        out = attn_prob @ value

        return out, attn_prob

class MultiHeadAttention(Model):
    def __init__(self, n_heads, dropout):
        assert d_model % n_heads == 0, 'd_model must be a multiple of n_heads.'

        self.dropout = dropout
        self.n_heads = n_heads

        self.pwq = Parameter()
        self.pwk = Parameter()
        self.pwv = Parameter()
        self.pwo = Parameter()
        self.attention = ScaledDotProductAttention(dropout)
        self.scan_attributes()

    def init(self, d_model):
        self.pwq.init([d_model, d_model], I.XavierUniform())
        self.pwk.init([d_model, d_model], I.XavierUniform())
        self.pwv.init([d_model, d_model], I.XavierUniform())
        self.pwo.init([d_model, d_model], I.XavierUniform())

    def __call__(self, query, key, value, mask, train):
        wq = F.parameter(self.pwq)
        wk = F.parameter(self.pwk)
        wv = F.parameter(self.pwv)
        wo = F.parameter(self.pwo)

        d_model = wq.shape()[0]
        d_k = d_model // self.n_heads

        seq_len = query.shape[0]
        query = F.reshape(query @ wq, (seq_len, self.n_heads, d_k))
        key = F.reshape(key @ wk, (seq_len, self.n_heads, d_k))
        value = F.reshape(value @ wv, (seq_len, self.n_heads, d_k))

        # TODO: should use F.split when it will be implemented.
        # shape = (n_heads, seq_len, d_k)
        query = F.concat([F.slice(query, 1, d_k * i, d_k * (i + 1)) for i in range(self.n_heads)], 2)
        key = F.concat([F.slice(key, 1, d_k * i, d_k * (i + 1)) for i in range(self.n_heads)], 2)
        value = F.concat([F.slice(value, 1, d_k * i, d_k * (i + 1)) for i in range(self.n_heads)], 2)

        heads, attn_prob = self.attention(d_k, query, key, value, mask, train)

        return F.reshape(heads, (self.seq_len, d_model)) @ wo

class PositionwiseFeedForward(Model):
    def __init__(self, dropout):
        self.dropout = dropout

        self.pw1 = Parameter()
        self.pb1 = Parameter()
        self.pw2 = Parameter()
        self.pb2 = Parameter()
        self.scan_attributes()

    def init(self, d_model, d_ff):
        self.pw1.init([d_ff, d_model], I.XavierUniform())
        self.pb1.init([d_ff], I.XavierUniform())
        self.pw2.init([d_model, d_ff], I.XavierUniform())
        self.pb2.init([d_model], I.XavierUniform())

    def __call__(self, x, train):
        w1 = F.parameter(self.pw1)
        b1 = F.parameter(self.pb1)
        w2 = F.parameter(self.pw2)
        b2 = F.parameter(self.pb2)

        h = F.dropout(F.relu(x @ w1 + b1), self.dropout, train)
        return h @ w2 + b2

class TransformerEncoderLayer(Model):
    def __init__(self, n_heads, dropout):
        self.dropout = dropout

        self.self_attention = MultiHeadAttention(n_heads, dropout)
        self.attn_norm = LayerNorm()
        self.feed_forward = PositionwiseFeedForward(dropout)
        self.ff_norm = LayerNorm()

        self.scan_attributes()

    def init(self, d_model, d_ff):
        self.self_attention.init(d_model)
        self.attn_norm.init(d_model)
        self.feed_forward.init(d_model, d_ff)
        self.ff_norm.init(d_model)

    def __call__(self, x, mask, train):
        attn = F.dropout(self.self_attention(x, x, x, mask, train), self.dropout, train)
        attn_res = self.attn_norm(x + attn)
        ff = F.dropout(self.feed_forward(attn_res, train), self.dropout, train)
        return self.ff_norm(attn_res + ff)

class TransformerEncoder(Model):
    def __init__(self, n_heads, n_stacks, dropout):
        self.layers = []
        for idx in range(n_stacks):
            layer = TransformerEncoderLayer(n_heads, dropout)
            self.add("encoder_layer" + str(idx), layer)
            self.layers.append(layer)

    def init(self, d_model, d_ff):
        for layer in self.layers:
            layer.init(d_model, d_ff)

    def __call__(self, src, mask, train):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoderLayer(Model):
    def __init__(self, n_heads, dropout):
        self.dropout = dropout

        self.self_attention = MultiHeadAttention(n_heads, dropout)
        self.self_attn_norm = LayerNorm()
        self.attention = MultiHeadAttention(n_heads, dropout)
        self.attn_norm = LayerNorm()
        self.feed_forward = PositionwiseFeedForward(dropout)
        self.ff_norm = LayerNorm()
        self.scan_attributes()

    def init(self, d_model, d_ff):
        self.self_attention.init(d_model)
        self.self_attn_norm.init(d_model)
        self.attention.init(d_model)
        self.attn_norm.init(d_model)
        self.feed_forward.init(d_model, d_ff)
        self.ff_norm.init(d_model)

    def __call__(self, x, src, src_mask, trg_mask, train):
        self_attn = F.dropout(self.self_attention(x, x, x, trg_mask, train), self.dropout, train)
        self_attn_res = self.self_attn_norm(x + self_attn)

        attn = F.dropout(self.attention(src, src, self_attn_res, train), self.dropout, train)
        attn_res = self.attn_norm(self_attn_res + attn)

        ff = F.dropout(self.feed_forward(attn_res, train), self.dropout, train)
        return self.ff_norm(attn_res + ff)

class TransformerDecoder(Model):
    def __init__(self, n_heads, n_stacks, dropout):
        self.layers = []
        for idx in range(n_stacks):
            layer = TransformerDecoderLayer(n_heads, dropout)
            self.add("decoder_layer" + str(idx), layer)
            self.layers.append(layer)

    def init(self, d_model, d_ff):
        for layer in self.layers:
            layer.init(d_model, d_ff)

    def __call__(self, trg, src, src_mask, trg_mask, train):
        for layer in self.layers:
            x = layers(x, src, src_mask, trg_mask, train)
        return x

class TransformerEmbeddings(Model):
    def __init__(self, dropout, max_len):
        self.max_len = max_len
        self.dropout = dropout
        self.pe = None

        self.plookup = Parameter()
        self.scan_attributes()

    def init(self, vocab, d_model):
        self.plookup.init([d_model, vocab], I.XavierUniform())

    def __call__(self, seq, train):

        lookup = F.parameter(self.plookup)
        if self.pe is None:
            self.pe = F.input(self.positional_encoding())

        embed = []
        for w in seq:
            e = F.pick(lookup, w)
            e *= math.sqrt(e.shape()[-1])
            pe = F.dropout(e + self.pe[:len(e)], self.dropout, train)
            embed.append(pe)
        return F.concat(embed, 2)

    def positional_encoding(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.expand_dims(np.arange(0, self.max_len), axis=1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        div_term = np.expand_dims(div_term, axis=0)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

class Transformer(Model):
    def __init__(self, dropout=0.1, max_len=5000):
        self.dropout = dropout
        self.max_len = max_len

        self.src_embed = TransformerEmbeddings(dropout, max_len)
        self.trg_embed = TransformerEmbeddings(dropout, max_len)
        self.encoder = TransformerEncoder(d_model, d_ff, n_heads, n_stacks, dropout)
        self.decoder = TransformerDecoder(d_model, d_ff, n_heads, n_stacks, dropout)
        self.scan_attributes()

    def init(self, vocab=37000, d_model=512, d_ff=2048, n_heads=8, n_stacks=6):
        self.src_embed.init(vocab, d_model)
        self.trg_embed.init(vocab, d_model)

    def encode(self, src, src_mask, train=True):
        return self.encoder(self.src_embed(src, train=train),
                            src_mask,
                            train=train)

    def decode(self, src, trg, src_mask, trg_mask, train=True):
        return self.decoder(self.trg_embed(trg, train=train),
                            src,
                            src_mask,
                            trg_mask,
                            train=train)

    def forward(self, src, tgt, src_mask, trg_mask, train=True):
        return self.decode(self.encode(src, src_mask, train=train),
                           trg,
                           src_mask,
                           trg_mask,
                           train=train)

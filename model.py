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
        self.pgain.init([1, d_model], I.Constant(1))
        self.pbias.init([1, d_model], I.Constant(0))

    def __call__(self, x):
        seq_len = x.shape()[0]
        gain = F.broadcast(F.parameter(self.pgain), 0, seq_len)
        bias = F.broadcast(F.parameter(self.pbias), 0, seq_len)

        dim = x.shape().depth()
        mean = F.mean(x, dim)
        std = F.sqrt(F.mean(x * x, dim) - mean * mean + self.eps)

        return gain * (x - mean) / std + bias

class ScaledDotProductAttention():
    def __init__(self, dropout):
        self.dropout = dropout

    def __call__(self, d_k, query, key, value, mask, train):
        attn = (query @ F.transpose(key)) / math.sqrt(d_k)

        if mask is not None:
            mask = F.input(mask)
            if attn.shape() != mask.shape():
                mask = F.broadcast(mask, 0, attn.shape()[0])
            attn -= 2000 * mask

        attn_prob = F.dropout(F.softmax(attn, 1), self.dropout, train)
        out = attn_prob @ value

        return out, attn_prob

class MultiHeadAttention(Model):
    def __init__(self, n_heads, dropout):
        self.dropout = dropout
        self.n_heads = n_heads

        self.pwq = Parameter()
        self.pwk = Parameter()
        self.pwv = Parameter()
        self.pwo = Parameter()
        self.attention = ScaledDotProductAttention(dropout)
        self.scan_attributes()

    def init(self, d_model):
        assert d_model % self.n_heads == 0, 'd_model must be a multiple of n_heads.'

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

        query_len = query.shape()[0]
        query = F.reshape(query @ wq, Shape([query_len, self.n_heads, d_k]))
        key_len = key.shape()[0]
        key   = F.reshape(key   @ wk, Shape([key_len,   self.n_heads, d_k]))
        value_len = value.shape()[0]
        value = F.reshape(value @ wv, Shape([value_len, self.n_heads, d_k]))

        query = [F.reshape(F.slice(query, 1, i, i + 1), Shape([query_len, d_k]))
                 for i in range(self.n_heads)]
        key   = [F.reshape(F.slice(key,   1, i, i + 1), Shape([key_len,   d_k]))
                 for i in range(self.n_heads)]
        value = [F.reshape(F.slice(value, 1, i, i + 1), Shape([value_len, d_k]))
                 for i in range(self.n_heads)]

        heads = []
        for q, k, v in zip(query, key, value):
            head, _ = self.attention(d_k, q, k, v, mask, train)
            heads.append(head)
        heads = F.concat(heads, 2)

        return F.reshape(heads, Shape([query_len, d_model])) @ wo

class PositionwiseFeedForward(Model):
    def __init__(self, dropout):
        self.dropout = dropout

        self.pw1 = Parameter()
        self.pb1 = Parameter()
        self.pw2 = Parameter()
        self.pb2 = Parameter()
        self.scan_attributes()

    def init(self, d_model, d_ff):
        self.pw1.init([d_model, d_ff], I.XavierUniform())
        self.pb1.init([1, d_ff], I.XavierUniform())
        self.pw2.init([d_ff, d_model], I.XavierUniform())
        self.pb2.init([1, d_model], I.XavierUniform())

    def __call__(self, x, train):
        seq_len = x.shape()[0]
        w1 = F.parameter(self.pw1)
        w2 = F.parameter(self.pw2)
        b1 = F.broadcast(F.parameter(self.pb1), 0, seq_len)
        b2 = F.broadcast(F.parameter(self.pb2), 0, seq_len)

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

    def __call__(self, src, mask, train):
        attn = F.dropout(self.self_attention(src, src, src, mask, train), self.dropout, train)
        attn_res = self.attn_norm(src + attn)
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
        x = src
        for layer in self.layers:
            x = layer(x, mask, train)
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

    def __call__(self, src, trg, src_mask, trg_mask, train):
        self_attn = F.dropout(self.self_attention(trg, trg, trg, trg_mask, train), self.dropout, train)
        self_attn_res = self.self_attn_norm(trg + self_attn)

        attn = F.dropout(self.attention(self_attn_res, src, src, src_mask, train), self.dropout, train)
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

        self.pwhy = Parameter()
        self.pby = Parameter()
        self.scan_attributes()

    def init(self, d_model, d_ff, vocab):
        for layer in self.layers:
            layer.init(d_model, d_ff)

        self.pwhy.init([d_model, vocab], I.XavierUniform())
        self.pby.init([1, vocab], I.XavierUniform())

    def __call__(self, src, trg, src_mask, trg_mask, train):
        h = trg
        for layer in self.layers:
            h = layer(src, h, src_mask, trg_mask, train)

        why = F.parameter(self.pwhy)
        by = F.broadcast(F.parameter(self.pby), 0, h.shape()[0])
        return h @ why + by

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
        d_model = lookup.shape()[0]
        if self.pe is None:
            self.pe = self.positional_encoding()

        embed = []
        for w in seq:
            e = F.pick(lookup, w, 1)
            embed.append(e)
        embed_tensor = F.transpose(F.concat(embed, 1))

        embed_tensor *= math.sqrt(d_model)
        pos = F.input(self.pe[:len(seq)])
        pe = F.dropout(embed_tensor + pos, self.dropout, train)
        return pe

    def positional_encoding(self):
        d_model = self.plookup.shape()[0]
        pe = np.zeros((self.max_len, d_model))
        position = np.expand_dims(np.arange(0, self.max_len), axis=1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        div_term = np.expand_dims(div_term, axis=0)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

class Transformer(Model):
    def __init__(self, n_heads=8, n_stacks=6, dropout=0.1, max_len=5000):
        self.n_heads = n_heads
        self.n_stacks = n_stacks
        self.dropout = dropout
        self.max_len = max_len

        self.src_embed = TransformerEmbeddings(dropout, max_len)
        self.trg_embed = TransformerEmbeddings(dropout, max_len)
        self.encoder = TransformerEncoder(n_heads, n_stacks, dropout)
        self.decoder = TransformerDecoder(n_heads, n_stacks, dropout)
        self.scan_attributes()

    def init(self, vocab=37000, d_model=512, d_ff=2048):
        self.src_embed.init(vocab, d_model)
        self.trg_embed.init(vocab, d_model)
        self.encoder.init(d_model, d_ff)
        self.decoder.init(d_model, d_ff, vocab)

    def encode(self, src, src_mask, train=True):
        return self.encoder(self.src_embed(src, train=train),
                            src_mask,
                            train=train)

    def decode(self, src, trg, src_mask, trg_mask, train=True):
        return self.decoder(src,
                            self.trg_embed(trg, train=train),
                            src_mask,
                            trg_mask,
                            train=train)

    def loss(self, src, trg, src_mask, trg_mask, train=True):
        output = self.decode(self.encode(src, src_mask, train=train),
                             trg[:-1],
                             src_mask,
                             trg_mask,
                             train=train)
        losses = []
        for i, t in enumerate(trg[1:]):
            y = F.reshape(F.pick(output, [i], 0), Shape([output.shape()[1]]))
            loss = F.softmax_cross_entropy(y, t, 0)
            losses.append(loss)
        loss = F.batch.mean(F.sum(losses))
        return loss

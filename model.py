# coding: utf-8

import math
import numpy as np

import primitiv
from primitiv import Parameter, Model, Shape
from primitiv import initializers as I

def function_type(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        train = kwargs['train'] if 'train' in kwargs else args[-1]
        self.F = primitiv.functions if train else primitiv.tensor_functions
        return func(*args, **kwargs)
    return wrapper

class LayerNorm(Model):
    def __init__(self, eps=1e-6):
        self.eps = eps

        self.pgain = Parameter()
        self.pbias = Parameter()
        self.scan_attributes()

    def init(self, d_model):
        self.pgain.init([1, d_model], I.Constant(1))
        self.pbias.init([1, d_model], I.Constant(0))

    @function_type
    def __call__(self, x, train):
        seq_len = x.shape()[0]
        d_model = x.shape()[1]

        gain = self.F.broadcast(self.F.parameter(self.pgain), 0, seq_len)
        bias = self.F.broadcast(self.F.parameter(self.pbias), 0, seq_len)

        mean = self.F.mean(x, 1)
        std = self.F.sqrt(self.F.mean(x * x, 1) - mean * mean)

        mean = self.F.broadcast(self.F.reshape(mean, Shape([seq_len, 1])), 1, d_model)
        std = self.F.broadcast(self.F.reshape(std, Shape([seq_len, 1])), 1, d_model)
        return gain * (x - mean) / (std + self.eps) + bias

class ScaledDotProductAttention():
    def __init__(self, dropout):
        self.dropout = dropout

    @function_type
    def __call__(self, query, key, value, mask, train):
        d_k = query.shape()[1]
        attn = (query @ self.F.transpose(key)) / math.sqrt(d_k) # [query_len, key_len]

        if mask is not None:
            mask = self.F.input(mask) # [1, key_len] or [query_len, key_len]
            if attn.shape() != mask.shape():
                mask = self.F.broadcast(mask, 0, attn.shape()[0])
            attn -= 2000 * mask

        attn_prob = self.F.dropout(self.F.softmax(attn, 1), self.dropout, train) # [query_len, key_len]
        out = attn_prob @ value # [query_len, d_k]

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

    @function_type
    def __call__(self, query, key, value, mask, train):
        wq = self.F.parameter(self.pwq)
        wk = self.F.parameter(self.pwk)
        wv = self.F.parameter(self.pwv)
        wo = self.F.parameter(self.pwo)

        d_model = wq.shape()[0]
        d_k = d_model // self.n_heads

        query_len = query.shape()[0]
        query = query @ wq
        query = [self.F.slice(query, 1, i * d_k, (i + 1) * d_k)
                 for i in range(self.n_heads)] # [n_heads, query_len, d_k]
        key_len = key.shape()[0]
        key   = key   @ wk
        key   = [self.F.slice(key  , 1, i * d_k, (i + 1) * d_k)
                 for i in range(self.n_heads)] # [n_heads, key_len, d_k]
        value_len = value.shape()[0]
        value = value @ wv
        value = [self.F.slice(value, 1, i * d_k, (i + 1) * d_k)
                 for i in range(self.n_heads)] # [n_heads, value_len, d_k]

        heads = []
        for q, k, v in zip(query, key, value):
            head, _ = self.attention(q, k, v, mask, train)
            heads.append(head)
        heads = self.F.concat(heads, 1)

        return heads @ wo

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

    @function_type
    def __call__(self, x, train):
        seq_len = x.shape()[0]
        w1 = self.F.parameter(self.pw1)
        w2 = self.F.parameter(self.pw2)
        b1 = self.F.broadcast(self.F.parameter(self.pb1), 0, seq_len)
        b2 = self.F.broadcast(self.F.parameter(self.pb2), 0, seq_len)

        h = self.F.dropout(self.F.relu(x @ w1 + b1), self.dropout, train)
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

    @function_type
    def __call__(self, src, mask, train):
        attn = self.F.dropout(self.self_attention(src, src, src, mask, train), self.dropout, train)
        attn_res = self.attn_norm(src + attn, train)
        ff = self.F.dropout(self.feed_forward(attn_res, train), self.dropout, train)
        return self.ff_norm(attn_res + ff, train)

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

    @function_type
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

    @function_type
    def __call__(self, src, trg, src_mask, trg_mask, train):
        self_attn = self.F.dropout(self.self_attention(trg, trg, trg, trg_mask, train), self.dropout, train)
        self_attn_res = self.self_attn_norm(trg + self_attn, train)

        attn = self.F.dropout(self.attention(self_attn_res, src, src, src_mask, train), self.dropout, train)
        attn_res = self.attn_norm(self_attn_res + attn, train)

        ff = self.F.dropout(self.feed_forward(attn_res, train), self.dropout, train)
        return self.ff_norm(attn_res + ff, train)

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

    @function_type
    def __call__(self, src, trg, src_mask, trg_mask, train):
        h = trg
        for layer in self.layers:
            h = layer(src, h, src_mask, trg_mask, train)

        why = self.F.parameter(self.pwhy)
        by = self.F.broadcast(self.F.parameter(self.pby), 0, h.shape()[0])
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

    @function_type
    def __call__(self, seq, train):
        lookup = self.F.parameter(self.plookup)
        d_model = lookup.shape()[0]
        if self.pe is None:
            self.pe = self.positional_encoding()

        embed = []
        for w in seq:
            e = self.F.pick(lookup, w, 1)
            embed.append(e)
        embed_tensor = self.F.transpose(self.F.concat(embed, 1))

        embed_tensor *= math.sqrt(d_model)
        pos = self.F.input(self.pe[:len(seq)])
        pe = self.F.dropout(embed_tensor + pos, self.dropout, train)
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

        self.embed = TransformerEmbeddings(dropout, max_len)
        self.encoder = TransformerEncoder(n_heads, n_stacks, dropout)
        self.decoder = TransformerDecoder(n_heads, n_stacks, dropout)
        self.scan_attributes()

    def init(self, vocab=37000, d_model=512, d_ff=2048):
        self.embed.init(vocab, d_model)
        self.encoder.init(d_model, d_ff)
        self.decoder.init(d_model, d_ff, vocab)

    def encode(self, src, src_mask, train=True):
        return self.encoder(self.embed(src, train=train),
                            src_mask,
                            train=train)

    def decode(self, src, trg, src_mask, trg_mask, train=True):
        return self.decoder(src,
                            self.embed(trg, train=train),
                            src_mask,
                            trg_mask,
                            train=train)

    @function_type
    def loss(self, src, trg, src_mask, trg_mask, train=True):
        output = self.decode(self.encode(src, src_mask, train=train),
                             trg[:-1],
                             src_mask,
                             trg_mask,
                             train=train)
        losses = []
        for i, t in enumerate(trg[1:]):
            y = self.F.reshape(self.F.pick(output, [i], 0), Shape([output.shape()[1]]))
            loss = self.F.softmax_cross_entropy(y, t, 0)
            losses.append(loss)
        loss = self.F.batch.mean(self.F.sum(losses))
        return loss

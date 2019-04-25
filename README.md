# Transformer primitiv

[Attention Is All You Need (Vaswani et al., NIPS'17)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
without beam search

it may work.  
2 stack, 100000 iterations, softmax cross-entropy on WMT En-De → BLEU = 21.46 pt

## Requirements
* Python >= 3.4
* [primitiv/primitiv-python v0.3.0](https://github.com/primitiv/primitiv-python/tree/v0.3.0)
* [google/sentencepiece](https://github.com/google/sentencepiece)
* tqdm
* numpy

## Usage
```sh
python main.py preproc [config file]
python main.py train [config file]
python main.py test [config file] > /path/to/output/file
```

# Reference
* [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

from collections import Counter


class Vocab(object):

    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    EOS_TOKEN = '<eos>'
    NUL_TOKEN = '<nul>'

    PAD_ID = 0
    UNK_ID = 1
    EOS_ID = 2
    NUL_ID = 3

    def __init__(self, vocab):
        self.idx2word = [Vocab.PAD_TOKEN, Vocab.UNK_TOKEN, Vocab.EOS_TOKEN,
                         Vocab.NUL_TOKEN]
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}
        for i, (word, _) in enumerate(vocab.most_common(), len(self.idx2word)):
            self.idx2word.append(word)
            self.word2idx[word] = i

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.i2w(key)
        if isinstance(key, str):
            return self.w2i(key)
        raise TypeError('key must be either int or str but %s' % type(key))

    def __len__(self):
        return len(self.word2idx)

    def w2i(self, word, allow_unk=False):
        if allow_unk:
            return self.word2idx.get(word, Vocab.UNK_ID)
        return self.word2idx[word]

    def ws2is(self, words, pad=False, max_len=None, allow_unk=False):
        result = [self.w2i(word, allow_unk) for word in words]
        if pad:
            num_pads = max_len - len(words)
            result += [self.PAD_ID] * num_pads
        return result

    def wss2iss(self, sents, pad=False, max_len=None, allow_unk=False):
        return [self.ws2is(sent, pad, max_len, allow_unk) for sent in sents]

    def i2w(self, idx):
        return self.idx2word[idx]

    def is2ws(self, idxs):
        return [self.i2w(idx) for idx in idxs]

    def iss2wss(self, idxss):
        return [self.is2ws(idxs) for idxs in idxss]


def select_vocab(vocab_size: int, vocab: Counter) -> Counter:
    total = sum(vocab.values())
    selected_vocab = Counter(dict(vocab.most_common(vocab_size)))
    selected = sum(selected_vocab.values())
    return selected_vocab

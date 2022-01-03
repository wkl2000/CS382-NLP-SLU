#coding=utf8
import os, json
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'


class Vocab():

    def __init__(self, padding=False, unk=False, min_freq=1, vocab_path=None, 
                spoken_language_select='asr_1best'):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK
        if vocab_path is not None:
            self.from_train(vocab_path, spoken_language_select, min_freq=min_freq)

    def from_train(self, vocab_path, spoken_language_select='asr_1best', min_freq=1):
        if isinstance(vocab_path, str):   # filepath : str | list([str, str, ...])
            vocab_path = [vocab_path]
        if isinstance(spoken_language_select, str):   # spoken_language_select : str | list([str, str]) & str in ['asr_1best', 'manual_transcript']
            spoken_language_select = [spoken_language_select]
        
        word_freq = {}
        for path in vocab_path:
            with open(path, 'r') as f:
                datas = json.load(f)
            for data in datas:
                for utt in data:
                    for sls in spoken_language_select:
                        text = utt[sls]
                        # TODO: there are some special sentences in manual_transcript
                        # if sls == "manual_transcript" and ("(unknown)" in text or "(side)" in text or "(dialect)" in text or "(noise)" in text or "(robot)" in text):
                        # TODO: there are some special sentences in asr_1best
                        # if sls == "asr_1best" and text == "null"
                        for char in text:
                            word_freq[char] = word_freq.get(char, 0) + 1
        
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])


class LabelVocab():

    def __init__(self, ontology_path):
        self.tag2idx, self.idx2tag = {}, {}

        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(ontology_path)

    def from_filepath(self, ontology_path):
        # ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r'))
        ontology = json.load(open(ontology_path, 'r'))
        
        acts = ontology['acts']
        slots = ontology['slots']

        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)

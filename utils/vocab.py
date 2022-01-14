#coding=utf8
import os, json
from xpinyin import Pinyin
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
        
        
        self.acts = ontology['acts']
        self.slots = ontology['slots']
        p = Pinyin()

        # print ("acts = ", self.acts)
        # print ("slots = ", self.slots)
        # print ("len(acts) = ", len(self.acts))
        # print ("len(slots) = ", len(self.slots))
        """
            RECALL THE MAP_DICT AND PINYIN_SET FOR THE DENOISING 
        """
        self.poi_map_dic, self.poi_pinyin_set = self.load_pinyin_from_file("./data/lexicon/poi_name.txt")  
        self.opera_map_dic, self.opera_pinyin_set = self.load_pinyin_from_file("./data/lexicon/operation_verb.txt")
        self.ordinal_map_dic, self.ordinal_pinyin_set = self.load_pinyin_from_file("./data/lexicon/ordinal_number.txt")

        self.request_map_dic, self.request_pinyin_set = self.load_pinyin_from_slots("请求类型")
        self.travel_map_dic, self.travel_pinyin_set = self.load_pinyin_from_slots("出行方式")
        self.route_map_dic, self.route_pinyin_set = self.load_pinyin_from_slots("路线偏好")
        self.object_map_dic, self.object_pinyin_set = self.load_pinyin_from_slots("对象")
        self.page_map_dic, self.page_pinyin_set = self.load_pinyin_from_slots("页码")


        for act in self.acts:
            for slot in self.slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    def load_pinyin_from_slots(self, slot_name):
        '''
            LOAD THE PINYIN FROM THE 
        
        '''
        p = Pinyin()
        map_dic = {}
        pinyin_set = set()
        slot_length = len(self.slots[slot_name])

        for i in range(slot_length):
            tmp_value = self.slots[slot_name][i]
            tmp_pinyin = p.get_pinyin(tmp_value, ' ')
            pinyin_set.add (tmp_pinyin)
            map_dic[tmp_pinyin] = tmp_value 
        
        return map_dic, pinyin_set


    def load_pinyin_from_file (self, file_name):
        p = Pinyin()
        map_dic = {}
        pinyin_set = set()

        f = open(file_name,"r")
        lines = f.readlines()
        for line in lines :
            line = line.replace("\n", "")
            tmp_pinyin = p.get_pinyin(line, ' ')
            pinyin_set.add (tmp_pinyin)
            map_dic[tmp_pinyin] = line
    
        return map_dic, pinyin_set

    @property
    def num_tags(self):
        return len(self.tag2idx)
        
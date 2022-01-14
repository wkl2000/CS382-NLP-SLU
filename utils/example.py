import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.bert2vec import BertUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, vocab_path=None, ontology_path=None, word2vec_path=None, 
                        spoken_language_select = 'asr_1best', word_embedding = 'Word2vec'):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, vocab_path=vocab_path,
                                spoken_language_select=spoken_language_select)
            # load train vocabularies


        if word_embedding == 'Word2vec' :
            cls.word2vec = Word2vecUtils(word2vec_path)  
        if word_embedding == 'Bert' :
            cls.word2vec = BertUtils(None)
            # there is 9600 common words with embedding dimension 768
            # head word2vec-768.txt -n 1
        cls.label_vocab = LabelVocab(ontology_path)
            # used to determine the number of tags and the category of tags
            # there is 2(B/I) * 2(acts: inform/deny) * 18(slots: poi名称, poi修饰, ...) + 2(O/<pad>) = 74 kinds of tags
    
    @classmethod
    def load_dataset(cls, data_path, spoken_language_select='asr_1best'):
        """
        cls : Example()
        data_path : path(es) of dataset, str | list([str, str, ...])
        spoken_language_select : str | list([str, str]), str in ['asr_1best', 'manual_transcript']
        """
        if isinstance(data_path, str):
            data_path = [data_path]
        if isinstance(spoken_language_select, str):
            spoken_language_select = [spoken_language_select]
        
        examples = []
        for sls in spoken_language_select:
            # assert spoken_language_select in ['asr_1best', 'manual_transcript']
            assert sls in ['asr_1best', 'manual_transcript']
            for path in data_path:
                datas = json.load(open(path, 'r'))
                for data in datas:
                    for utt in data:
                        ex = cls(utt, sls)
                        examples.append(ex)
            
        return examples


    def __init__(self, ex: dict, spoken_language_select):
        super(Example, self).__init__()
        self.ex = ex
        """
        ex['utt_id'] : 对话的轮数
        ex['manual_transcript'] : 人工修正过的无噪音的输入
        ex['asr_1best'] : ASR 模型识别出来的结果
        ex['semantic'] : 语义标签
        For example (datas[0][0]), 
            {'utt_id': 1,
            'manual_transcript': '给我导航云南省昆明市黄土坡',
            'asr_1best': '导航云南省昆明市黄土坡',
            'semantic': [['inform', '操作', '导航'], ['inform', '终点名称', '云南省昆明市黄土坡']]}
        """
        # self.utt = ex['asr_1best']
        # self.utt = ex['manual_transcript']
        # assert spoken_language_select in ['asr_1best', 'manual_transcript']
        self.utt = ex[spoken_language_select]
        
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        
        self.input_idx = [Example.word_vocab[c] for c in self.utt]  
            # `Example.word_vocab.word2id : dict` turn vocab into index
                # {'<pad>': 0, '<unk>': 1, '导': 2, '航': 3, '云': 4, '南': 5, '省': 6, '昆': 7, ...}
            # len(Example.word_vocab.word2id) == 1782 when 
                # spoken_language_select = 'asr_1best'
                # and use train.json only
        self.tag_id = [Example.label_vocab.convert_tag_to_idx(tag) for tag in self.tags]
            # `Example.label_vocab.tag2idx : dict` turn tag into index
        """
        Output is stored in self.utt(vocab seq), self.tags(tag seq), self.input_idx(vocab seq index), self.tag_id(tag seq index)
        For example (datas[0][0]),
            self.utt = '导航云南省昆明市黄土坡'
            self.slot = {'inform-操作': '导航', 'inform-终点名称': '云南省昆明市黄土坡'}
            self.tags = [   'B-inform-操作',
                            'I-inform-操作',
                            'B-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称',
                            'I-inform-终点名称'     ]
            self.slotvalue = ['inform-操作-导航', 'inform-终点名称-云南省昆明市黄土坡']
            self.input_idx = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            self.tag_id = [30, 31, 14, 15, 15, 15, 15, 15, 15, 15, 15]
        """


    def __str__(self):
        return f"vocab seq: {self.utt}\ntag seq: {self.tags}\nvocab seq(index): {self.input_idx}\ntag seq(index): {self.tag_id}"

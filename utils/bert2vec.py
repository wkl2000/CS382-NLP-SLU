#coding=utf8

from utils.vocab import PAD, UNK
import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel


class BertUtils():

    def __init__(self, Bert_file):
        super(BertUtils, self).__init__()
        self.Bert = {}
        self.read_model()

    def load_embeddings(self, module, vocab, device='cpu'):
        """ Initialize the embedding with glove and char embedding
        """
        emb_size = module.weight.data.size(-1)
        outliers = 0

        for index, word in enumerate (vocab.word2id):
            if index % 500 == 0 :
                print (index)
            if word == PAD: # PAD symbol is always 0-vector
                module.weight.data[vocab[PAD]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            # use the bert to get the word embedding 
            inputs = self.tokenizer.encode_plus(word, padding="max_length", truncation=True, max_length=10,
                                        add_special_tokens=True,
                                        return_tensors="pt")     
            out = self.bert_model(**inputs)
            word_emb = out[0][0][0].detach().numpy()

            module.weight.data[vocab[word]] = torch.tensor(word_emb, dtype=torch.float, device=device)
        return 1 - outliers / float(len(vocab))

    def read_model(self):
        model_name = 'bert-base-chinese'
        MODEL_PATH = './chinese_wwm_ext_pytorch'
 
        # a.通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # b. 导入配置文件
        model_config = BertConfig.from_pretrained(model_name)
        # 修改配置
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        # 通过配置和路径导入模型
        self.bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)

        # inputs = tokenizer.encode_plus('大', padding="max_length", truncation=True, max_length=10,
        #                             add_special_tokens=True,
        #                             return_tensors="pt")
        # out = bert_model(**inputs)
        # out = out[0][0][0].detach().numpy().tolist()
        # print(out)
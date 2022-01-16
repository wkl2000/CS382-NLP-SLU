#coding=utf8
# 测试脚本：读取./data/test_unlabelled.json中的文件，将结果写回到./data/test_unlabelled.json中

import sys, os, time, gc
from unittest import TextTestResult
from torch.optim import Adam
from math import sqrt
import pickle
from xpinyin import Pinyin
from datetime import datetime
import json

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging
from tqdm import tqdm




def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer

def decode(model, dataset, device, args):
    # assert choice in ['train', 'dev']
    model.eval()
    # dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        if args.anti_noise == True:
            predictions = anti_noise_prediction(predictions)
        metrics = Example.evaluator.acc(predictions, labels)
    # torch.cuda.empty_cache()
        # RuntimeError: CUDA error: out of memory
        # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
    gc.collect()
    return metrics, total_loss / count

def anti_noise_prediction(predictions):
    p = Pinyin()
    select_pos_set = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}
    select_others_set = {'请求类型': [Example.label_vocab.request_map_dic, Example.label_vocab.request_pinyin_set], \
        '出行方式' : [Example.label_vocab.travel_map_dic, Example.label_vocab.travel_pinyin_set], \
        '路线偏好' : [Example.label_vocab.route_map_dic, Example.label_vocab.route_pinyin_set], \
        '对象' :  [Example.label_vocab.object_map_dic, Example.label_vocab.object_pinyin_set], \
        '页码' : [Example.label_vocab.page_map_dic, Example.label_vocab.page_pinyin_set], \
        '操作' : [Example.label_vocab.opera_map_dic, Example.label_vocab.opera_pinyin_set], \
        '序列号' : [Example.label_vocab.ordinal_map_dic, Example.label_vocab.ordinal_pinyin_set]   }

    modify_num = 0
    for i, pred in enumerate(predictions):
        pred_length = len(pred)
        if pred_length > 0 :
            for j in range(pred_length):
                tmp_pred = pred[j]
                split_result = tmp_pred.split('-')
                tmp_pinyin = p.get_pinyin(split_result[2], ' ')
                if split_result[1] != 'value' :
                    if split_result[1] in select_pos_set :
                        map_dic, pinyin_set = Example.label_vocab.poi_map_dic, Example.label_vocab.poi_pinyin_set
                    else :
                        [map_dic, pinyin_set] = select_others_set[split_result[1]]

                    standard_output = get_standard_output (map_dic, pinyin_set, tmp_pinyin)
                    modify_pred = split_result[0] + '-' + split_result[1] + '-' + standard_output
                    if standard_output != split_result[2] :
                        modify_num += 1
                    # print ("standard_output = ", standard_output, " split_result[2] = ", split_result[2])

                    # print ("modify_pred = ", modify_pred, " predictions[i][j] = ", predictions[i][j])
                    predictions[i][j] = modify_pred
    # print ("modify_num == ", modify_num)                    
    return  predictions            

def get_standard_output (map_dic, pinyin_set, tmp_pinyin) :
    if tmp_pinyin in pinyin_set :
        standard_output = map_dic[tmp_pinyin]
    else :
        max_similarity = 0
        most_similar_pinyin = ''
        for standard_pinyin in iter(pinyin_set) :
            similarity = get_pinyin_similarity(standard_pinyin, tmp_pinyin)
            if similarity > max_similarity :
                max_similarity = similarity
                most_similar_pinyin = standard_pinyin
        if max_similarity == 0 : 
            standard_output = '无'
        else :
            standard_output = map_dic[most_similar_pinyin]
    return standard_output
            

def get_pinyin_similarity(standard_pinyin, tmp_pinyin) :
    standard_set = set (standard_pinyin.split(' '))
    tmp_set = set (tmp_pinyin.split(' '))

    inter_set = standard_set & tmp_set
    similarity = len (inter_set) / (len (standard_set) + len (tmp_set) )
    return similarity



def evaluate(model, dataset, device, args):
    # start_time = time.time()
    metrics, dev_loss = decode(model, dataset, device, args)
        # metrics, dev_loss = decode('dev')
        # dataset = train_dataset if choice == 'train' else dev_dataset
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    return dev_loss, dev_acc, dev_fscore


def preperation(args):

    # load configuration
    start_time = time.time()
    train_path = os.path.join(args.dataroot, 'train.json')
    dev_path = os.path.join(args.dataroot, 'development.json')
    ontology_path = os.path.join(args.dataroot, 'ontology.json')
    word2vec_path = args.word2vec_path
    if args.trainset_spoken_language_select == "both":
        args.trainset_spoken_language_select = ['asr_1best', 'manual_transcript']
    if args.trainset_augmentation:
        aug_path = os.path.join(args.dataroot, 'augmentation.json')
        train_path = [train_path, aug_path]
    else:
        train_path = train_path
    Example.configuration(  vocab_path=train_path, 
                            ontology_path=ontology_path, 
                            word2vec_path=word2vec_path,
                            spoken_language_select=args.trainset_spoken_language_select,
                            word_embedding = args.word_embedding)
    
    # load dataset and preprocessing
    # train_dataset = Example.load_dataset(train_path)
    train_dataset = Example.load_dataset(train_path, spoken_language_select=args.trainset_spoken_language_select)
    dev_dataset = Example.load_dataset(dev_path, spoken_language_select='asr_1best')
    print("Load dataset and database finished, cost %.2f s ..." % (time.time() - start_time))
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    # update some parameters based on corpus
    args.vocab_size = Example.word_vocab.vocab_size
    args.pad_idx = Example.word_vocab[PAD]
    args.num_tags = Example.label_vocab.num_tags
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
        # changes will be stored in `args`

    # model
    device = set_torch_device(args.device)
    print("Use cuda:%s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

    return Example, train_dataset, dev_dataset, device


if __name__ == "__main__":
    args = init_args(sys.argv[1:])
    print("Initialization finished ...")

    Example, train_dataset, dev_dataset, device = preperation(args)
    model = SLUTagging(args).to(device)
    if not os.path.exists(args.best_model_dir):
        print("无法load模型!")
        exit(-1)
        
    checkpoint = torch.load(args.best_model_dir)
    model.load_state_dict(checkpoint['model'])
    
    print("Devset Evaluation : ", end="\n\t")
    dev_loss, dev_acc, dev_fscore = evaluate(model, dev_dataset, device, args)

    # 下面开始计算在./data/test_unlabelled.json上进行inference的结果
    
    with open(args.unlabeled_data_path, 'r') as load_f:
        test_unlabelled = json.load(load_f)


    test_dataset = Example.load_dataset(args.unlabeled_data_path, spoken_language_select='asr_1best')
    model.eval()
    predictions = []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            # print(f"cur_dataset:{len(cur_dataset)}")
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            
            total_loss += loss
            count += 1
        if args.anti_noise == True:
            predictions = anti_noise_prediction(predictions)
        
    gc.collect()
    # print(predictions)

    test_labelled = test_unlabelled
    # 下面把predictions对应地填回到test_labelled中
    i=0
    for j,group in enumerate(test_unlabelled):
        for k,each in enumerate(group): # 对应每一句话
            # print(each["asr_1best"],predictions[i])
            for slot_str in predictions[i]:
               test_labelled[j][k]["pred"].append(slot_str.split('-'))
            i+=1

with open(args.labeled_data_path, 'w',encoding="utf-8") as load_f:
        json.dump(test_labelled,load_f,indent=4,ensure_ascii=False)
print(f"对{args.unlabeled_data_path}的预测结果已经保存到{args.labeled_data_path}中")


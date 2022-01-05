#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    # arg_parser.add_argument('--seed', default=2021, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')

    arg_parser.add_argument('--results_dir', default='./results', help='path for saving results')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=50, help='terminate after maximum epochs')

    arg_parser.add_argument('--eval_interval', default=1, type=int, help='number of intervals to evaluate')
    arg_parser.add_argument('--restore', action='store_true', help='restore training if a checkpoint exists.')
    arg_parser.add_argument('--save_best_model', action='store_false', help='restore training if a checkpoint exists.')
    arg_parser.add_argument('--checkpoint_interval', default=5, type=int, help='number of intervals to save a checkpoint(<0 for no checkpoint)')
    arg_parser.add_argument('--checkpoint_dir', default='./scripts/checkpoint.bin', help='path of checkpoint of model')
    arg_parser.add_argument('--best_model_dir', default='./scripts/model.bin', help='path of best model')

    arg_parser.add_argument('--trainset_spoken_language_select', default='asr_1best', choices=['manual_transcript', 'asr_1best', 'both'], 
                            help='*sentence used for trainset(asr_1best: with noise; manual_transcript: without noise)')
    arg_parser.add_argument('--trainset_augmentation', action='store_true', help='*used augmented data from lexicon')
    arg_parser.add_argument('--early_stop_epoch', type=int, default=10, help='number of epochs to check early stop(<0 for no early stop)')

    arg_parser.add_argument('--runs', type=int, default=1, help='run multiple times to get more accurate evaluation results')
    
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='*root of data')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--rnn_hidden_size', default=512, type=int, help='hidden size for rnn')
    arg_parser.add_argument('--rnn_num_layers', default=2, type=int, help='number of layer for rnn')
    arg_parser.add_argument('--mlp_hidden_size', default=256, type=int, help='hidden size for mlp')
    arg_parser.add_argument('--mlp_num_layers', default=1, type=int, help='*number of layer for mlp')
    
    return arg_parser
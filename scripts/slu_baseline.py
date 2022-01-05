#coding=utf8
import sys, os, time, gc
from torch.optim import Adam
from math import sqrt
import pickle

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

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
        metrics = Example.evaluator.acc(predictions, labels)
    # torch.cuda.empty_cache()
        # RuntimeError: CUDA error: out of memory
        # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
    gc.collect()
    return metrics, total_loss / count

def evaluate(model, dataset, device, args):
    # start_time = time.time()
    metrics, dev_loss = decode(model, dataset, device, args)
        # metrics, dev_loss = decode('dev')
        # dataset = train_dataset if choice == 'train' else dev_dataset
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    return dev_loss, dev_acc, dev_fscore


def train(model, train_dataset, dev_dataset, device, args):
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'iter':0, 'dev_loss': 100.0, 'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    start_epoch = 0

    # load checkpoint
    if args.restore and os.path.exists(args.checkpoint_dir):
        print(f'Load checkpoint from "{args.checkpoint_dir}" ')
        checkpoint = torch.load(args.checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optim'])        # can not match!
        start_epoch = checkpoint['epoch'] + 1

    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * (args.max_epoch - start_epoch)
    # print('Total training steps: %d' % (num_training_steps))
    
    # print('Start training ......')
    dev_acc_history = []
    dev_loss_history = []
    for epoch in tqdm(range(start_epoch, args.max_epoch)):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        # print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (epoch, time.time() - start_time, epoch_loss / count))
        print('Training: \tEpoch: %d\tLoss: %.4f' % (epoch, epoch_loss / count))
        # torch.cuda.empty_cache()
            # RuntimeError: CUDA error: out of memory
            # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
        gc.collect()
        
        # save checkpoint
        if args.checkpoint_interval > 0 and (epoch+1) % args.checkpoint_interval == 0:
            checkpoint = {   'epoch': epoch, 'model': model.state_dict(),   }
            torch.save(checkpoint, args.checkpoint_dir)
        
        # evaluate model with dev_dataset
        if args.eval_interval <= 0 or (epoch+1) % args.eval_interval == 0:
            print(f"Epoch: {epoch+1}", end="\t")
            dev_loss, dev_acc, dev_fscore = evaluate(model, dev_dataset, device, args)
            dev_acc_history.append(round(dev_acc, 2))
            dev_loss_history.append(round(dev_loss, 2))

            # best acc model
            if args.save_best_model and dev_acc > best_result['dev_acc']:
                print(f'Save best model to "{args.best_model_dir}" ')
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, epoch
                best_model = {   'epoch': epoch, 'model': model.state_dict(),   }
                torch.save(best_model, args.best_model_dir)
                # print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (epoch, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

            # early stop
            if args.early_stop_epoch > 0 and epoch > args.early_stop_epoch:
                """
                avg_dev_loss = sum(dev_loss_history[-(args.early_stop_epoch+1):-1]) / args.early_stop_epoch
                if dev_loss > avg_dev_loss:     # dev loss degenerating means overfitting, then early stop is needed!
                    print(f'Early Stoped! Total Epochs: {epoch}')
                    break
                """
                # use dev_acc instead of dev_loss as early stop criterion
                avg_dev_acc = sum(dev_acc_history[-(args.early_stop_epoch+1):-1]) / args.early_stop_epoch
                if dev_loss < avg_dev_acc:     # dev loss degenerating means overfitting, then early stop is needed!
                    print(f'Early Stoped! Total Epochs: {epoch}')
                    break
    
    print("-------------------- Train Result --------------------")
    print(f"Dev Acc: {dev_acc_history}")
    print(f"Dev Loss: {dev_loss_history}")
    print('BEST RESULT: \tDev loss: %.2f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    print("-------------------- Train Result --------------------")

    return best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']


def train_wrap(args, Example, train_dataset, dev_dataset, device):
    # set random seed
    set_random_seed(args.seed)
    print("\nRandom seed is set to %d" % (args.seed))
    
    # initialize model
    model = SLUTagging(args).to(device)
    Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)
        # Initialize `model.word_embed` by `Example.word_vocab`
    
    # train model
    dev_loss, dev_acc, dev_fscore = train(model, train_dataset, dev_dataset, device, args)
    
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
                            spoken_language_select=args.trainset_spoken_language_select)
    
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
    # initialization args
    args = init_args(sys.argv[1:])
    print("Initialization finished ...")

    Example, train_dataset, dev_dataset, device = preperation(args)

    
    if not args.testing:
        # some fixed seeds for training
        # SEEDs = [518030910376, 518021911048, 518021911033, 2021, 2018, 1896, 200240]
            # 5180xxxxxxxx is lager than 2**32 - 1, while seed must be between 0 and 2**32 - 1 ...
        SEEDs = [30910376, 21911048, 21911033, 2022, 2021, 2020, 2019, 2018, 1896, 200240]
        
        # get model results
        results = []
        for run_i in range(args.runs):
            args.seed = SEEDs[run_i]
            dev_loss, dev_acc, dev_fscore = train_wrap(args, Example, train_dataset, dev_dataset, device)
            result = {"loss": dev_loss, "acc": dev_acc, "fscore": dev_fscore}
            results.append(result)
        
        # save results
        config_str = f"rnn-{args.encoder_cell}_mlp-{args.mlp_num_layers}_aug-{args.trainset_augmentation:d}_train-{args.trainset_spoken_language_select}"
        save_path = os.path.join(args.results_dir, config_str + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        """
        load_path = f".pkl"
        with open(load_path, "rb") as f:
            results = pickle.load(f)
        """

        # print accuracy results
        print(f"========== {config_str} ==========")
        dev_acc_results = [result["acc"] for result in results]
        print(f"DEV ACC = {dev_acc_results}")
        avg = sum(dev_acc_results) / len(dev_acc_results)
        std = sqrt(sum([(r-avg)**2 for r in dev_acc_results]) / len(dev_acc_results))
        print(f"DEV ACC = {avg} ± {std}")
        print(f"========== {config_str} ==========\n")

    else:
        model = SLUTagging(args).to(device)
        # Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)       # Initialize `model.word_embed` by `Example.word_vocab`
        # load best model
        if os.path.exists(args.best_model_dir):
            print(f'Load checkpoint from "{args.best_model_dir}" ')
            checkpoint = torch.load(args.best_model_dir)
            model.load_state_dict(checkpoint['model'])

        print("Devset Evaluation", end="\n\t")
        dev_loss, dev_acc, dev_fscore = evaluate(model, dev_dataset, device, args)

        print("Trainset Evaluation", end="\n\t")
        train_loss, train_acc, train_fscore = evaluate(model, train_dataset, device, args)

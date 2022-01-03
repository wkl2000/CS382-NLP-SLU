#coding=utf8
import sys, os, time, gc
from torch.optim import Adam

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
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

def evaluate(model, dataset, device, args):
    start_time = time.time()
    metrics, dev_loss = decode(model, dataset, device, args)
        # metrics, dev_loss = decode('dev')
        # dataset = train_dataset if choice == 'train' else dev_dataset
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

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
    print('Total training steps: %d' % (num_training_steps))
    print('Start training ......')
    for i in tqdm(range(start_epoch, args.max_epoch)):
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
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()
        
        # save checkpoint
        if (i+1) % args.checkpoint_interval == 0:
            checkpoint = {   'epoch': i, 'model': model.state_dict(),   }
            torch.save(checkpoint, args.checkpoint_dir)
        
        # evaluate model with dev_dataset
        if (i+1) % args.eval_interval == 0:
            print(f"Epoch: {i+1}", end="\t")
            dev_loss, dev_acc, dev_fscore = evaluate(model, dev_dataset, device, args)
        
            # best acc model
            if dev_acc > best_result['dev_acc']:
                print(f'Save best model to "{args.best_model_dir}" ')
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
                best_model = {   'epoch': i, 'model': model.state_dict(),   }
                torch.save(best_model, args.best_model_dir)
                print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))


if __name__ == "__main__":
    # initialization params, output path, logger, random seed and torch.device
    args = init_args(sys.argv[1:])
    set_random_seed(args.seed)
    device = set_torch_device(args.device)
    print("Initialization finished ...")
    print("Random seed is set to %d" % (args.seed))
    print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

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
    print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    # update some parameters based on corpus
    args.vocab_size = Example.word_vocab.vocab_size
    args.pad_idx = Example.word_vocab[PAD]
    args.num_tags = Example.label_vocab.num_tags
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

    # model 
    model = SLUTagging(args).to(device)
    Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)       # Initialize `model.word_embed` by `Example.word_vocab`

    if not args.testing:
        train(model, train_dataset, dev_dataset, device, args)
    else:
        # load best model
        if os.path.exists(args.best_model_dir):
            print(f'Load checkpoint from "{args.best_model_dir}" ')
            checkpoint = torch.load(args.best_model_dir)
            model.load_state_dict(checkpoint['model'])

        print("Devset Evaluation", end="\n\t")
        evaluate(model, dev_dataset, device, args)

        print("Trainset Evaluation", end="\n\t")
        evaluate(model, train_dataset, device, args)
# SJTU CS382 SLU

本项目为上海交通大学 CS382 NLP 课程的课程大作业三：口语语义理解任务(SLU)
本小组成员为：张泽熙、丁立、吴凯龙

### 创建环境

```
conda create -n slu python=3.6
source activate slu
pip install torch==1.7.1
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==3.4.0
pip install xpinyin
```

### 运行

在根目录下运行

```
python scripts/slu_baseline.py
python scripts/slu_baseline.py --max_epoch 10 --device 3    # 训练模型，最大 epoch 数为 10，使用 3 号 GPU 卡
python scripts/slu_baseline.py --restore --max_epoch 10 --device 3    # 训练模型，如果存在 checkpoint 则恢复训练
python scripts/slu_baseline.py --testing --device 3   # 测试训练模型，使用 3 号 GPU 卡
python scripts/slu_baseline.py --max_epoch 10 --device 3 --trainset_spoken_language_select both   # 同时使用有噪声和无噪声的数据进行训练
python scripts/slu_baseline.py --max_epoch 10 --device 3 --trainset_spoken_language_select both --trainset_augmentation   # 同时使用有噪声和无噪声的数据以及 lexicon 增强的数据进行训练
python scripts/slu_baseline.py --max_epoch 10 --device 3 --trainset_spoken_language_select both --trainset_augmentation --encoder_cell GRU  # 使用 GRU 作为 RNN 的 cell/block
python scripts/slu_baseline.py --max_epoch 10 --device 3 --trainset_spoken_language_select both --trainset_augmentation --encoder_cell GRU --mlp_num_layers 2   # 设置 MLP 的层数为 2
python scripts/slu_baseline.py --max_epoch 50 --early_stop_epoch 10 --device 3 --trainset_spoken_language_select both --trainset_augmentation --encoder_cell GRU --mlp_num_layers 2   # early stop 设置为 10
python scripts/slu_baseline.py --max_epoch 50 --early_stop_epoch 10 --device 3 --trainset_spoken_language_select both --trainset_augmentation --encoder_cell GRU --mlp_num_layers 2 --runs 2  # 重复运行 10 次以更好地评估模型
```

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成

  ```
    python scripts/slu_baseline.py --<arg> <value>
  ```

  其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
  
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU

+ `utils/vocab.py`:构建编码输入输出的词表

+ `utils/word2vec.py`:读取Word2vec词向量

+ `utils/bert2vec.py`:由bert预训练模型得到词向量

+ `utils/example.py`:读取数据

+ `utils/batch.py`:将数据以批为单位转化为输入

+ `model/slu_baseline_tagging.py`:baseline模型

+ `scripts/slu_baseline.py`:主程序脚本

+ `chinese_wwm_ext_pytorch`:使用的bert预训练模型，来源：https://github.com/ymcui/Chinese-BERT-wwm

### 有关预训练语言模型

本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型

+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库

+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
+ nltk
  + 强力的NLP工具库: https://www.nltk.org/
+ stanza
  + 强力的NLP工具库: https://stanfordnlp.github.io/stanza/
+ jieba
  + 中文分词工具: https://github.com/fxsjy/jieba

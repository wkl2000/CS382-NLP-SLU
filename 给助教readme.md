# SJTU CS382 SLU

本项目为上海交通大学 CS382 NLP 课程的课程大作业三：口语语义理解任务(SLU)
本小组成员为：吴凯龙、丁立、张泽熙

### 创建环境


```
conda create -n slu python=3.6
source activate slu
pip install torch==1.7.1
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==3.4.0
pip install xpinyin

```

### 运行
首先进入项目文件夹，然后运行以下命令，其中--unlabeled_data_path参数可以进行修改，改为想要测试的文件的路径，运行完inference后，结果也会存到相同目录下。
其他参数不要修改！
```
python scripts/test.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation     --trainset_spoken_language_select manual_transcript --anti_noise --word_embedding Bert --unlabeled_data_path=./data/test_unlabelled.json
```

### 代码说明
主要测试逻辑在`scripts/test.py`中


### 联系
如果整个代码运行过程有任何问题，请及时联系
张泽熙 手机：18730966056

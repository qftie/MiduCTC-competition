## 生成最后提交结果
在command目录下执行`sh test_gen.sh`

## **数据说明**

### 数据文件

> 数据文件都为json格式，可使用json读取加载
- `data/final_train_fusion_stage2_3.json`: 用初赛的extend，valid和决赛的extend，valid数据合并得到的stage2的训练数据。
其中合并数据集生成新数据集的代码写在`notebook/data_extend_generate.ipynb`中

### json文件字段说明

- id: 文本id
- source: 源文本（可能包含错误的文本）
- target: 目标文本（正确文本）
- type: positive代表正样本， negative代表负样本

## 模型说明
### 模型

以**GECToR**作为baseline模型，可参考[GECToR论文](https://aclanthology.org/2020.bea-1.16.pdf)和[GECToR源代码](https://github.com/grammarly/gector)
backbone则替换为了`hfl/chinese-macbert-base`
### 代码结构
```
MiduCTC-competition
├─ command
│  ├─ test_gen.sh # 提交文件生成脚本
│  └─ train.sh # 训练脚本
├─ data
│  ├─ .gitkeep
│  ├─ example.txt
│  ├─ example_input.json
│  ├─ example_output.json
│  └─ final_train_fusion_stage2_3.json # 融合后的训练数据
├─ notebook
│  ├─ DifflibTest.ipynb
│  ├─ data_analy.ipynb
│  └─ data_extend_generate.ipynb # 利用初赛和决赛数据合成用来多折训练的数据
├─ requirements.txt
├─ src
│  ├─ __init__.py
│  ├─ baseline
│  │  ├─ __init__.py
│  │  ├─ ctc_vocab
│  │  │  ├─ config.py
│  │  │  ├─ ctc_correct_tags.txt
│  │  │  └─ ctc_detect_tags.txt
│  │  ├─ dataset.py
│  │  ├─ loss.py
│  │  ├─ modeling.py
│  │  ├─ predictor.py # 包含利用seq2edit的模型生成修改后句子的函数，考虑了多个模型
│  │  ├─ tokenizer.py
│  │  └─ trainer.py
│  ├─ corrector.py
│  ├─ evaluate.py
│  ├─ kfolds_prepare_for_uploads.py # 生成提交文件的入口，可以用多折训练的模型融合生成
│  ├─ metric.py # 指标计算文件
│  ├─ prepare_for_upload.py
│  └─ train.py # 训练入口，可以设置要分几折，以及需要训练的特定折数
└─ test_output.json # 生成的提交文件

```


## 训练说明
该模型训练按GECToR的论文所述，尝试两个stage和三个stage的训练方法，由于验证下来两个stage显著优于只用伪数据训练，而三个stage相对两个stage提升不大，所以选择了两个stage的训练方式。

### Stage1
第一个stage先在100w条样本的伪数据上进行训练，将训练得到的在`preliminary_val.json`上效果最优的权重作为stage2的预训练权重。这里直接将第一个stage训练得到的权重等文件保存在`pretrained_model/ch_macbert_base_epoch5,step1,testf1_39_41%,devf1_67_26%`,方便stage2的调用。

### Stage2
第二个stage使用`pretrained_model/ch_macbert_base_epoch5,step1,testf1_39_41%,devf1_67_26%`作为预训练权重，使用合并的初赛和决赛数据合并的`data/final_train_fusion_stage2_3.json`数据集，分为十折来进行训练和验证，最后选取的是验证集表现最好的两组权重平均考虑其预测，生成最后得分`Fscore=51.89`的提交文件。

要复现结果，在command目录下执行`sh train.sh`,然后执行`sh test_gen.sh`,参数可以在sh文件中修改，

- trian.sh文件
```sh
cd .. && export CUDA_DEVICE_ORDER=PCI_BUS_ID && CUDA_VISIBLE_DEVICES=0,1,2,5 python -m src.train \
--in_model_dir "pretrained_model/ch_macbert_base_epoch5,step1,testf1_39_41%,devf1_67_26%" \
--out_model_dir "model/ctc_train" \
--epochs "10" \
--batch_size "64" \
--max_seq_len "256" \
--learning_rate "4e-5" \
--train_fp "data/final_train_fusion_stage2_3.json" \
--random_seed_num "43" \
--check_val_every_n_epoch "0.5" \
--early_stop_times "20" \
--warmup_steps "100" \
--dev_data_ratio "0.2" \
--training_mode "dp" \
--amp true \
--freeze_embedding false
```

- test_gen.sh文件
```sh
cd .. && export CUDA_DEVICE_ORDER=PCI_BUS_ID && CUDA_VISIBLE_DEVICES=5 python -m src.kfolds_prepare_for_uploads \
--model4upload_dir 'model/ctc_train_2022Y08M14D15H/epoch8,step51,testf1_63_29%,devf1_51_68%/' \
--in_test_json_file 'change in_file model in src/kfolds_prepare_for_uploads.py'
```

**权重选择**：两组最好的权重如下，另外请注意，定义使用哪些权重进行融合预测的代码是直接写在`src/kfolds_prepare_for_uploads.py`中的，要改权重需要到文件中去改

```python
in_model_dirs=[
        'model/ctc_train_2022Y08M18D11H/fold_2/epoch5,step1,testf1_None,devf1_54_24%/',
        'model/ctc_train_2022Y08M18D12H/fold_6/epoch5,step35,testf1_None,devf1_52_37%',
        ]
```

## 调优和trick搜索
### trick
在a榜b榜的提交过程中尝试了不同的trick均未有明显提升所以最后没有使用其他trick（尝试过的trick有迭代纠错、使用detect输出判断整句话是否有错，如果最大检错概率小于一定的阈值则认为该句没有出错直接跳过，测试记录可见提交结果记录文档）

### backbone
在stage1尝试过roberta-base、macbert-base、pert-base、macbert-large，调优后发现macbert-base效果较好，个人觉得应该是因为macbert预训练就是使用了错字或者span替换等策略和gec中出现最多的错误类似，pert则是使用的语序打乱复原的预训练方式，可能对于乱序的错误的错误更有效果，也有考虑融合不同模型的优势，但由于时间问题没有尝试，但不清楚为什么large大模型反而效果更差，也许是因为没有足够的计算资源尝试lr调优


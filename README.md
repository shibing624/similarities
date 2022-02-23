[![PyPI version](https://badge.fury.io/py/similarities.svg)](https://badge.fury.io/py/similarities)
[![Downloads](https://pepy.tech/badge/similarities)](https://pepy.tech/project/similarities)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/similarities.svg)](https://github.com/shibing624/similarities/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/similarities.svg)](https://github.com/shibing624/similarities/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# Similarities
Similarities is a toolkit for Compute Similarity Score between texts. 

相似度计算工具包，实现多种字面、语义匹配模型。

**similarities**实现了Word2Vec、RankBM25、BERT、Sentence-BERT、CoSENT等多种文本表征、文本相似度计算模型，并在文本语义匹配（相似度计算）任务上比较了各模型的效果。


**Guide**
- [Feature](#Feature)
- [Evaluate](#Evaluate)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Citation](#Citation)
- [Reference](#reference)

# Feature
### 文本向量表示模型
- [Word2Vec](similarities/word2vec.py)：通过腾讯AI Lab开源的大规模高质量中文[词向量数据（800万中文词轻量版）](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) (文件名：light_Tencent_AILab_ChineseEmbedding.bin 密码: tawe）实现词向量检索，本项目实现了句子（词向量求平均）的word2vec向量表示
- [SBert(Sentence-BERT)](similarities/sentence_bert)：权衡性能和效率的句向量表示模型，训练时通过有监督训练上层分类函数，文本匹配预测时直接句子向量做余弦，本项目基于PyTorch复现了Sentence-BERT模型的训练和预测
- [CoSENT(Cosine Sentence)](similarities/cosent)：CoSENT模型提出了一种排序的损失函数，使训练过程更贴近预测，模型收敛速度和效果比Sentence-BERT更好，本项目基于PyTorch实现了CoSENT模型的训练和预测

### 文本相似度比较方法

- 余弦相似（Cosine Similarity）：两向量求余弦
- 点积（Dot Product）：两向量归一化后求内积
- 词移距离（Word Mover’s Distance）：词移距离使用两文本间的词向量，测量其中一文本中的单词在语义空间中移动到另一文本单词所需要的最短距离
- [RankBM25](similarities/bm25.py)：BM25的变种算法，对query和文档之间的相似度打分，得到docs的rank排序
- [SemanticSearch](https://github.com/shibing624/similarities/blob/master/similarities/sbert.py#L80)：向量相似检索，使用Cosine Similarty + topk高效计算，比一对一暴力计算快一个数量级

# Evaluate

### 文本匹配

- 英文匹配数据集的评测结果：

| Arch | Backbone | Model Name | English-STS-B | 
| :-- | :--- | :--- | :-: |
| GloVe | glove | Avg_word_embeddings_glove_6B_300d | 61.77 |
| BERT | bert-base-uncased | BERT-base-cls | 20.29 |
| BERT | bert-base-uncased | BERT-base-first_last_avg | 59.04 |
| BERT | bert-base-uncased | BERT-base-first_last_avg-whiten(NLI) | 63.65 |
| SBERT | sentence-transformers/bert-base-nli-mean-tokens | SBERT-base-nli-cls | 73.65 |
| SBERT | sentence-transformers/bert-base-nli-mean-tokens | SBERT-base-nli-first_last_avg | 77.96 |
| SBERT | xlm-roberta-base | paraphrase-multilingual-MiniLM-L12-v2 | 84.42 |
| CoSENT | bert-base-uncased | CoSENT-base-first_last_avg | 69.93 |
| CoSENT | sentence-transformers/bert-base-nli-mean-tokens | CoSENT-base-nli-first_last_avg | 79.68 |

- 中文匹配数据集的评测结果：

| Arch | Backbone | Model Name | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg | QPS |
| :-- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| CoSENT | hfl/chinese-macbert-base | CoSENT-macbert-base | 50.39 | **72.93** | **79.17** | **60.86** | **80.51** | **68.77**  | 2572 |
| CoSENT | Langboat/mengzi-bert-base | CoSENT-mengzi-base | **50.52** | 72.27 | 78.69 | 12.89 | 80.15 | 58.90 | 2502 |
| CoSENT | bert-base-chinese | CoSENT-bert-base | 49.74 | 72.38 | 78.69 | 60.00 | 80.14 | 68.19 | 2653 |
| SBERT | bert-base-chinese | SBERT-bert-base | 46.36 | 70.36 | 78.72 | 46.86 | 66.41 | 61.74 | 1365 |
| SBERT | hfl/chinese-macbert-base | SBERT-macbert-base | 47.28 | 68.63 | **79.42** | 55.59 | 64.82 | 63.15 | 1948 |
| CoSENT | hfl/chinese-roberta-wwm-ext | CoSENT-roberta-ext | **50.81** | **71.45** | **79.31** | **61.56** | **81.13** | **68.85** | - |
| SBERT | hfl/chinese-roberta-wwm-ext | SBERT-roberta-ext | 48.29 | 69.99 | 79.22 | 44.10 | 72.42 | 62.80 | - |

- 本项目release模型的中文匹配评测结果：

| Arch | Backbone | Model Name | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg | QPS |
| :-- | :--- | :---- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Word2Vec | word2vec | w2v-light-tencent-chinese | 20.00 | 31.49 | 59.46 | 2.57 | 55.78 | 33.86 | 10283 |
| SBERT | xlm-roberta-base | paraphrase-multilingual-MiniLM-L12-v2 | 18.42 | 38.52 | 63.96 | 10.14 | 78.90 | 41.99 | 2371 |
| CoSENT | hfl/chinese-macbert-base | similarities-base-chinese | 31.93 | 42.67 | 70.16 | 17.21 | 79.30 | **48.25** | 2572 |

说明：
- 结果值均使用spearman系数
- 结果均只用该数据集的train训练，在test上评估得到的表现，没用外部数据
- `paraphrase-multilingual-MiniLM-L12-v2`模型名称是`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，是`paraphrase-MiniLM-L12-v2`模型的多语言版本，速度快，效果好，支持中文
- `CoSENT-macbert-base`模型达到同级别参数量SOTA效果，是用CoSENT方法训练，运行[similarities/cosent](similarities/cosent)文件夹下代码可以复现结果
- `SBERT-macbert-base`模型，是用SBERT方法训练，运行[similarities/sentence_bert](similarities/sentence_bert)文件夹下代码可以复现结果
- `similarities-base-chinese`模型，是用CoSENT方法训练，基于MacBERT在中文STS-B数据训练得到，模型文件已经上传到huggingface的模型库[shibing624/similarities-base-chinese](https://huggingface.co/shibing624/similarities-base-chinese)
- `w2v-light-tencent-chinese`是腾讯词向量的Word2Vec模型，CPU加载使用
- 各预训练模型均可以通过transformers调用，如MacBERT模型：`--pretrained_model_path hfl/chinese-macbert-base`
- 中文匹配数据集下载[链接见下方](#数据集)
- 中文匹配任务实验表明，pooling最优是`first_last_avg`，预测可以调用SBert的`mean pooling`方法，效果损失很小
- QPS的GPU测试环境是Tesla V100，显存32GB

# Demo

Official Demo: http://42.193.145.218/product/short_text_sim/

HuggingFace Demo: https://huggingface.co/spaces/shibing624/similarities

![](docs/hf.png)

# Install
```
pip3 install -U similarities
```

or

```
git clone https://github.com/shibing624/similarities.git
cd similarities
python3 setup.py install
```

### 数据集
常见中文语义匹配数据集，包含[ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC)、[BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm)、[LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[PAWSX](https://arxiv.org/abs/1908.11828)、[STS-B](https://github.com/pluto-junzeng/CNSD)共5个任务。
可以从数据集对应的链接自行下载，也可以从[百度网盘(提取码:qkt6)](https://pan.baidu.com/s/1d6jSiU1wHQAEMWJi7JJWCQ)下载。

其中senteval_cn目录是评测数据集汇总，senteval_cn.zip是senteval目录的打包，两者下其一就好。

# Usage

### 1. 计算文本向量


### 2. 计算句子之间的相似度值

示例[semantic_text_similarity.py](./examples/semantic_text_similarity.py)


> 句子余弦相似度值`score`范围是[-1, 1]，值越大越相似。

### 3. 计算句子与文档集之间的相似度值

一般在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配、文本相似检索等任务。



> `Score`的值范围[-1, 1]，值越大，表示该query与corpus相似度越近。



# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/similarities.svg)](https://github.com/shibing624/similarities/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了similarities，请按如下格式引用：

```latex
@misc{similarities,
  title={similarities: A Tool for Compute Similarity Score},
  author={Ming Xu},
  howpublished={https://github.com/shibing624/similarities},
  year={2022}
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加similarities的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [A Simple but Tough-to-Beat Baseline for Sentence Embeddings[Sanjeev Arora and Yingyu Liang and Tengyu Ma, 2017]](https://openreview.net/forum?id=SyK00v5xx)
- [四种计算文本相似度的方法对比[Yves Peirsman]](https://zhuanlan.zhihu.com/p/37104535)
- [谈谈文本匹配和多轮检索](https://zhuanlan.zhihu.com/p/111769969)

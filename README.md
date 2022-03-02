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

### 文本相似度比较方法

- 余弦相似（Cosine Similarity）：两向量求余弦
- 点积（Dot Product）：两向量归一化后求内积
- 词移距离（Word Mover’s Distance）：词移距离使用两文本间的词向量，测量其中一文本中的单词在语义空间中移动到另一文本单词所需要的最短距离
- [RankBM25](similarities/bm25.py)：BM25的变种算法，对query和文档之间的相似度打分，得到docs的rank排序
- [SemanticSearch](https://github.com/shibing624/similarities/blob/master/similarities/sbert.py#L80)：向量相似检索，使用Cosine Similarty + topk高效计算，比一对一暴力计算快一个数量级

# Evaluate

### 文本匹配

# Demo

Official Demo: http://42.193.145.218/product/short_text_sim/

HuggingFace Demo: https://huggingface.co/spaces/shibing624/text2vec

![](docs/hf.png)

# Install
```
pip3 install torch # conda install pytorch
pip3 install -U similarities
```

or

```
git clone https://github.com/shibing624/similarities.git
cd similarities
python3 setup.py install
```


# Usage

### 1. 计算句子之间的相似度值

示例[semantic_text_similarity.py](./examples/semantic_text_similarity.py)


> 句子余弦相似度值`score`范围是[-1, 1]，值越大越相似。

### 2. 计算句子与文档集之间的相似度值

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

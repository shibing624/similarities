[![PyPI version](https://badge.fury.io/py/similarities.svg)](https://badge.fury.io/py/similarities)
[![Downloads](https://pepy.tech/badge/similarities)](https://pepy.tech/project/similarities)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/similarities.svg)](https://github.com/shibing624/similarities/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/similarities.svg)](https://github.com/shibing624/similarities/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# Similarities
Similarities is a toolkit for similarity calculation and semantic search based on matching model. 

similarities：相似度计算、语义匹配搜索工具包。

**similarities**基于多种字面、语义匹配模型，实现了各模型的相似度计算、匹配搜索功能，python3开发，pip安装，开箱即用。


**Guide**
- [Feature](#Feature)
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
- [RankBM25](similarities/literalsim.py)：BM25的变种算法，对query和文档之间的相似度打分，得到docs的rank排序
- [SemanticSearch](https://github.com/shibing624/similarities/blob/main/similarities/similarity.py#L99)：向量相似检索，使用Cosine Similarty + topk高效计算，比一对一暴力计算快一个数量级


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

### 1. 计算两个句子的相似度值

```shell
from similarities import Similarity
m = Similarity("shibing624/text2vec-base-chinese")
r = m.similarity('如何更换花呗绑定银行卡', '花呗更改绑定银行卡')
print(f"{r:.4f}")
```

output:
```shell
0.8551
```

> 句子余弦相似度值`score`范围是[-1, 1]，值越大越相似。

### 2. 文档集中相似文本搜索

一般在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配、文本相似检索等任务。


中文示例[examples/base_demo.py](./examples/base_demo.py)

```python
from similarities import Similarity

if __name__ == '__main__':
    model = Similarity("shibing624/text2vec-base-chinese")
    # 1.Compute cosine similarity between two sentences.
    sentences = ['如何更换花呗绑定银行卡',
                 '花呗更改绑定银行卡']
    corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
        '俄罗斯警告乌克兰反对欧盟协议',
        '暴风雨掩埋了东北部；新泽西16英寸的降雪',
        '中央情报局局长访问以色列叙利亚会谈',
        '人在巴基斯坦基地的炸弹袭击中丧生',
    ]
    similarity_score = model.similarity(sentences[0], sentences[1])
    print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")
    
    # 2.Compute similarity between two list
    similarity_scores = model.similarity(sentences, corpus)
    print(similarity_scores.numpy())
    for i in range(len(sentences)):
        for j in range(len(corpus)):
            print(f"{sentences[i]} vs {corpus[j]}, score: {similarity_scores.numpy()[i][j]:.4f}")

    # 3.Semantic Search
    m = Similarity("shibing624/text2vec-base-chinese", corpus=corpus)
    q = '如何更换花呗绑定银行卡'
    print(m.most_similar(q, topn=5))
    print("query:", q)
    for i in m.most_similar(q, topn=5):
        print('\t', i)
```

output:
```shell
如何更换花呗绑定银行卡 vs 花呗更改绑定银行卡, score: 0.8551
...

如何更换花呗绑定银行卡 vs 花呗更改绑定银行卡, score: 0.8551
如何更换花呗绑定银行卡 vs 我什么时候开通了花呗, score: 0.7212
如何更换花呗绑定银行卡 vs 俄罗斯警告乌克兰反对欧盟协议, score: 0.1450
如何更换花呗绑定银行卡 vs 暴风雨掩埋了东北部；新泽西16英寸的降雪, score: 0.2167
如何更换花呗绑定银行卡 vs 中央情报局局长访问以色列叙利亚会谈, score: 0.2517
如何更换花呗绑定银行卡 vs 人在巴基斯坦基地的炸弹袭击中丧生, score: 0.0809
花呗更改绑定银行卡 vs 花呗更改绑定银行卡, score: 1.0000
花呗更改绑定银行卡 vs 我什么时候开通了花呗, score: 0.6807
花呗更改绑定银行卡 vs 俄罗斯警告乌克兰反对欧盟协议, score: 0.1714
花呗更改绑定银行卡 vs 暴风雨掩埋了东北部；新泽西16英寸的降雪, score: 0.2162
花呗更改绑定银行卡 vs 中央情报局局长访问以色列叙利亚会谈, score: 0.2728
花呗更改绑定银行卡 vs 人在巴基斯坦基地的炸弹袭击中丧生, score: 0.1279

query: 如何更换花呗绑定银行卡
	 (0, '花呗更改绑定银行卡', 0.8551459908485413)
	 (1, '我什么时候开通了花呗', 0.721195638179779)
	 (4, '中央情报局局长访问以色列叙利亚会谈', 0.2517135739326477)
	 (3, '暴风雨掩埋了东北部；新泽西16英寸的降雪', 0.21666759252548218)
	 (2, '俄罗斯警告乌克兰反对欧盟协议', 0.1450251191854477)
```
> `Score`的值范围[-1, 1]，值越大，表示该query与corpus的文本越相似。


英文示例[examples/base_english_demo.py](./examples/base_english_demo.py)


### 3. 快速近似匹配搜索

支持Annoy、Hnswlib的近似匹配搜索，常用于百万数据集的匹配搜索任务。


示例[examples/fast_sim_demo.py](./examples/fast_sim_demo.py)


### 4. 基于字面的文本相似度计算

支持同义词词林（Cilin）、知网Hownet、词向量（WordEmbedding）、Tfidf、Simhash、BM25等算法的相似度计算和匹配搜索，常用于文本匹配冷启动。

示例[examples/literal_sim_demo.py](./examples/literal_sim_demo.py)

```python
from similarities.literalsim import SimhashSimilarity, TfidfSimilarity, BM25Similarity, \
    WordEmbeddingSimilarity, CilinSimilarity, HownetSimilarity

text1 = "如何更换花呗绑定银行卡"
text2 = "花呗更改绑定银行卡"

m = TfidfSimilarity()
print(text1, text2, ' sim score: ', m.similarity(text1, text2))
print('distance:', m.distance(text1, text2))
zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
m.add_corpus(zh_list)
print(m.most_similar('刘若英是演员'))
```

output:
```shell
如何更换花呗绑定银行卡 花呗更改绑定银行卡  sim score:  0.8203384355246909
distance: 0.17966156447530912

[(0, '刘若英是个演员', 0.9847577834309504), (3, '我不是演员吗', 0.7056381915655814), (1, '他唱歌很好听', 0.5), (2, 'women喜欢这首歌', 0.5)]
```

# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/similarities.svg)](https://github.com/shibing624/similarities/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了similarities，请按如下格式引用：

APA:
```
Xu, M. Similarities: Compute similarity score for humans (Version 0.0.4) [Computer software]. https://github.com/shibing624/similarities
```

BibTeX:
```
@software{Xu_Similarities_Compute_similarity,
author = {Xu, Ming},
title = {Similarities: similarity calculation and semantic search toolkit},
url = {https://github.com/shibing624/similarities},
version = {0.0.4}
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
- [liuhuanyong/SentenceSimilarity](https://github.com/liuhuanyong/SentenceSimilarity)
- [shibing624/text2vec](https://github.com/shibing624/text2vec)
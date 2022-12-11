# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')

from similarities.literalsim import (
    SimHashSimilarity,
    TfidfSimilarity,
    BM25Similarity,
    WordEmbeddingSimilarity,
    CilinSimilarity,
    HownetSimilarity,
    SameCharsSimilarity,
    SequenceMatcherSimilarity,
)

from similarities.utils.distance import hamming_distance


class LiteralCase(unittest.TestCase):
    def test_hamming_distance(self):
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        r = 1.0 - hamming_distance(seq1, seq2) / 64
        print(hamming_distance(seq1, seq2))
        print(r)
        print(f"{text1} vs {text2} sim score {m.similarity(text1, text2)}")

        text1 = '刘若英是个演员'
        text2 = '他'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        print(f"{text1} vs {text2} sim score {m.similarity(text1, text2)}")

        text1 = '刘若英唱歌'
        text2 = '唱歌'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        print(f"{text1} vs {text2} sim score {m.similarity(text1, text2)}")

        text1 = '刘若'
        text2 = '刘若他他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听,他唱歌很好听?他唱歌很好听？他唱歌很好听。。'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        print(f"{text1} vs {text2} sim score {m.similarity(text1, text2)}")

        text1 = '刘若 他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听,他唱歌很好听?他唱歌很好听？他唱歌很好'
        text2 = '他他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听,他唱歌很好听?他唱歌很好听？他唱歌很好听。。'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        s = m.similarity(text1, text2)
        print(f"{text1} vs {text2} sim score {s}")
        self.assertTrue(s[0] > 0.5)

    def test_simhash(self):
        """test_simhash"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = SimHashSimilarity()
        print(f"{text1} vs {text2} sim score {m.similarity(text1, text2)}")
        print(m.distance(text1, text2))
        r = m.most_similar('刘若英是演员')
        self.assertEqual(len(r[0]), 0)
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=2)
        print(r)
        self.assertEqual(len(r[0]), 2)

    def test_short_text_simhash(self):
        text1 = '你妈妈喊你回家吃饭哦，回家罗回家罗'
        text2 = '你妈妈叫你回家吃饭哦，回家罗回家罗'
        m = SimHashSimilarity()
        seq1 = m.ori_simhash(text1)
        seq2 = m.ori_simhash(text2)
        print(seq1)
        print(seq2)
        scores = [m._sim_score(seq1, seq2) for seq1, seq2 in zip([seq1], [seq2])]
        print(f"{text1} vs {text2} ori_simhash sim score {scores}")

        def simhash_demo(text_a, text_b):
            """
            求两文本的相似度
            :param text_a:
            :param text_b:
            :return:
            """
            from simhash import Simhash
            a_simhash = Simhash(text_a)
            b_simhash = Simhash(text_b)
            print(a_simhash.value)
            max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
            # 汉明距离
            distince = a_simhash.distance(b_simhash)
            print(distince)
            similar = 1 - distince / max_hashbit
            return similar

        similar = simhash_demo(text1, text2)
        print(f"{text1} vs {text2} simhash_demo sim score {similar}")
        print(f"{text1} vs {text2} simhash sim score {m.similarity(text1, text2)}")

        text1 = "平台专注于游戏领域,多年的AI技术积淀,一站式提供文本、图片、音/视频内容审核,游戏AI以及数据平台服务"
        text2 = "平台专注于游戏领域,多年的AI技术积淀,二站式提供文本、图片、音 视频内容审核,游戏AI以及数据平台服务"
        text3 = '平台专注于游戏领域,多年的AI技术积淀,三站式提供文本、图片、音视频内容审核'
        similar = simhash_demo(text1, text2)
        similar2 = simhash_demo(text1, text3)
        similar3 = simhash_demo(text2, text3)
        print(similar)
        print(similar2)
        print(similar3)

        print(f"{text1} vs {text2} sim score {m.similarity(text1, text2)}")
        print(m.distance(text1, text2))
        r = m.most_similar('刘若英是演员')
        self.assertEqual(len(r[0]), 0)
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=2)
        print(r)

    def test_tfidf(self):
        """test_tfidf"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = TfidfSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员')
        print(r)
        self.assertEqual(len(r[0]), 4)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_bm25(self):
        """test_bm25"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = BM25Similarity()
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=10)
        print(r)
        self.assertEqual(len(r[0]), 4)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_word2vec(self):
        """test_word2vec"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        list_of_corpus = ["This is a test1", "This is a test2", "This is a test3"]
        list_of_corpus2 = ["that is test4", "that is a test5", "that is a test6"]
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '刘若英是个演员', '演戏很好看的人']
        m = WordEmbeddingSimilarity(list_of_corpus)
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        m.add_corpus(list_of_corpus2 + zh_list)
        v = m._get_vector("This is a test1")
        print(v[:10], v.shape)
        r = m.most_similar('刘若英是演员', topn=4)
        print(r)
        self.assertEqual(len(r[0]), 4)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_cilin(self):
        """test_cilin"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = CilinSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=3)
        print(r)
        self.assertEqual(len(r[0]), 3)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_hownet(self):
        """test_cilin"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = HownetSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=2)
        print(r)
        self.assertEqual(len(r[0]), 2)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_samechars(self):
        """test_samechars"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = SameCharsSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))

        text1 = '刘若英是演员'
        text2 = '刘若英是个演员'
        m = SameCharsSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=3)
        print(r)
        self.assertEqual(len(r[0]), 3)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_seqmatcher(self):
        """test_seqmatcher"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = SequenceMatcherSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))

        text1 = '刘若英是演员'
        text2 = '刘若英是个演员'
        m = SequenceMatcherSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=3)
        print(r)
        self.assertEqual(len(r[0]), 3)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)


if __name__ == '__main__':
    unittest.main()

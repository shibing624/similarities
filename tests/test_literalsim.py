# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')

from similarities.literal_similarity import (
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
        text1 = """
        是缘分还是注定，选择了你——威然。家里以前就有台大众途观，算起来开了也有七八年了，一直很稳当大问题也没出过。多数是老爸和我在开，随着去年年底我家臭小子降临，我用车时间比老爸用的时间多多了，顿时感觉家里一台车有点周转不过来啦。于是就有着增加一台车的想法，经过全家人讨论来讨论去，最终决定买台MPV，其实我内心一直是想要台SUV的，奈何以一敌三完败，老爸老妈的想法很奇葩，说是要为二胎考虑，没有那么钱总让我去换新车开，老婆最直接说MPV坐起来舒服些，她要享受老板级的待遇，这样一来也注定了我的司机命。      话说选车过程也是很艰辛啊，左挑右选，各种对比试驾，最后还是选择了大众，因为有第一台的感受，觉得大众还是很靠谱的，再一个就是从安全的角度出发，大众也绝对的让我放心。老婆又不太懂，只说威然坐起来真的非常舒服，就这样把它定了下来。。。      废话不多说了，趁着等我家老婆大人的空隙，给它搞了个写真，做好准备，我要上图咯前俩依然采用得是大众家族化得设计，大灯与前中网得完美融合使得前脸看起来很舒服，一体式得中网加上镀铬饰条得装饰让前脸看起来紧致又有张力，符合高端商务得气质Viloran作为一款全新车型，从外观来看突破了大众家族设计元素。从前脸来看新车新颖、大气，车头格栅与前大灯融为一体，而且进气格栅面积更大更宽，整体感进一步提高。前保险杠贯穿式通风口，内部由镀铬填充，侧面看会显得很长，整体得造型也是增加了很多得线条，看起来更加得丰满车尾部的造型比较中庸很符合大众的形象。尾灯做的也是比较大的，很有辨识度。大众VILORAN 的车标位置也显示出这是一辆豪华车。大众下面带字母都是豪车。内饰的设计也是特别的豪华，精致，液晶仪表盘，多媒体大屏幕应有尽有！中控也是全触摸的，基本没有按键。第二排的双杯架和手机之家，下面还有2个SUB充电口。很人性化的设计。坐在车里看着窗外的风景疾驰而过，心里真的非常惬意！！
        
        """
        text2 = """
        samples_1/J.txt 让生命不留遗憾，我选择——威然，提车作业！。就在上个星期左右，我们终于把威然给提了回来。在这个城市和老乡打拼多年，总算有了我们自己的一席之地，大学毕业后，我和老乡选择留在了这个快节奏而又充满着机遇和挑战的城市，在这个城市我们从初踏入职场，到联合创业，有过太多的心酸、也有太多的欢乐，我们始终坚信通过我们的努力定会在这个城市绽放出属于我们的精彩，在这个城市我们都邂逅了我们的挚爱，公司业务不忙的空档，我们两家也经常组团旅游，但很少自驾出去过，出于公司业务和家庭出游的需要，我和老乡商量着准备入手一台商务车，预算定在了三四十万，我们也知道这个价格开的GL8居多，但我们都感觉GL8太商务了，我们想买个宜商宜家的车，业务需要，有时会机场、高铁站接客户，所以对第二排要求一定要舒适、最好带按摩功能，这样我们的客户旅途劳顿可以在车上短暂的休息，无疑会提高我们公司服务形象，而且有时候带父母出游的话，第二排乘坐舒适对他们也比较好，自己平常也很喜欢车，也经常看一些车评，当时就看到了大众新上了一款大型的MPV威然，于是对他很好奇。威然给我的第一感觉就是酷、帅气,试驾的感觉也很棒。去了两次4S店，第二次是带着家人去的，买车不是件小事必须得征得媳妇的同意才行，我老婆见了这辆车之后也是很激动，能够看得出她那种喜悦之情。不仅是车子的外观吸引我，宽敞的空间对于我们家来说适合六口人乘坐，内饰的设计很大气高端有档次,说心里话我对她的感觉胜过了GL8，豪华感爆棚。所以最终还是把他提回来了。。图片已删除前俩依然采用得是大众家族化得设计，大灯与前中网得完美融合使得前脸看起来很舒服，一体式得中网加上镀铬饰条得装饰让前脸看起来紧致又有张力，符合高端商务得气质图片已删除Viloran作为一款全新车型，从外观来看突破了大众家族设计元素。从前脸来看新车新颖、大气，车头格栅与前大灯融为一体，而且进气格栅面积更大更宽，整体感进一步提高。前保险杠贯穿式通风口，内部由镀铬填充，图片已删除LED前大灯，炯炯有神图片已删除轮毂很漂亮图片已删除大红色尾灯，由银色饰条将两侧的尾灯相连，想不引人注意都不行图片已删除侧面看会显得很长，整体得造型也是增加了很多得线条，看起来更加得丰满图片已删除车尾部的造型比较中庸很符合大众的形象。尾灯做的也是比较大的，很有辨识度。大众VILORAN的车标位置也显示出这是一辆豪华车。大众下面带字母都是豪车。图片已删除威然这款车的后备箱超级大，同级别里最大的，不好的一点就是后排座椅和后备箱不平。图片已删除打开进入到主驾驶舱，一股豪华气息迎面而来。凡是平时接触比较多的位置都是采用了皮质的包裹，很舒服。而且这个浅色的内饰我个人也是非常喜欢。图片已删除多功能方向盘图片已删除第一眼就能看见这个全液晶的仪表盘，很有档次。图片已删除内饰的设计也是特别的豪华，精致，液晶仪表盘，多媒体大屏幕应有尽有！图片已删除中控也是全触摸的，基本没有按键。图片已删除杯架和杯架前面的都是储物空间。图片已删除扶手箱空间也蛮大的图片已删除图片已删除第二排的座椅，看上去就很舒服。也是电动调节的！图片已删除图片已删除第二排的双杯架和手机之家，下面还有2个SUB充电口。很人性化的设计。图片已删除门板非常厚实图片已删除图片已删除图片已删除第二排中央的空调出风口，还有座椅通风加热，都是很贴心的功能。图片已删除图片已删除全景大天窗，还有谁……图片已删除坐在车里看着窗外的风景疾驰而过，心里真的非常惬意！！
        """
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

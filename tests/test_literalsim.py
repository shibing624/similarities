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

    def test_bm25_hardcase(self):
        """test_bm25 hardcase"""
        m = BM25Similarity()
        zh_list = [
            "的可能。术后还需要坚持服用药物治疗。药物治疗建议采用以补益心肾、涤痰熄风,开窍定痫,活血化淤、平肝泻火的中药汤药进行治疗,调理身体状况,调理脏腑机能,固本培元,达到治愈的目的。中药副作用小,标本兼治，治愈后不易复发",
            "我是星期一做的子宫息肉手术是否有关系吗，女51岁因为最近有些不舒服，就去进行了检查，但是一查是这个病，请问：我是星期一做的子宫息肉手术是否有关系",
            "腥膻的食物，一定是不要喝酒的，还有就是一定要多喝水，可以多吃些蔬果。前列腺囊肿这个疾病的患者一定是要注意休息的，一定是要保证自己充足的睡眠，还有就是一定是不要熬夜的，也尽量是不要高强度地工作的，还有就是一定是不要给自己过多的心理压力的，要学会放松自己。掌握前列腺囊肿的症状表现能够及时的发现疾病，并且不会因为由于对疾病的不了解，从而产生忽视，以及错误治疗的情况，事实上，对于前列腺疾病来说，非常容易根据其症状表现",
            "汤含服，或者用青果，凤凰衣等常用药物进行煎服，有一定效果。，考虑是肌肉软组织疼痛，建议口服风疼片，或者舒筋丸治疗都可以的，另外如果看中医方便也可地中医，进行中药治疗效果很好，积极对症治疗，慢慢会改善的，祝你健康",
            '内科护理学的辅助检查有些什么？", "answer": "客观结构化临床考试"}\n{"question": "白消安注射液的用法用量',
            "为疾病的统称来说，则具有比较基本的症状表现，毕竟类型不同，但是有些方面则是相同的，另外，疾病的类型不同，主要就是根据病因所决定，这点需要明确，下面就前列腺囊肿的基本症状表现进行总结：第一、排尿异常，主要的表现是排尿的异常(尿频、尿急、尿等待、尿分叉、排尿无力)。有的会有射精无力的症状，但是早期不会有症状。第二、分泌异常。主要是前列腺液异常的分泌导致的，(排尿滴白、排便滴白、尿频",
            '但是一查是这个病，请问：我是星期一做的子宫息肉手术是否有关系吗", "answer": "宫颈息肉（很小）手术后需要有卧床休息,当然如果出血量并不多,可适当的去户外活动的,防止重体力劳动就可,防止增强腹压而致使大出血的再次发生.宫颈息肉是慢性宫颈炎的一种表现出,息肉的根部粘附于子宫颈口或宫颈管内,是宫颈粘膜在炎症的刺激下局部纤维化,并外突于宫颈外口而构成.由于息肉割除比较简单,对身体伤损性也不大',
            '对患者有一定的影响。所以患者应该注意自身护理。流产患者注意自身护理，例如当的进行饮食调理，注意保暖避免着凉。"}\n{"question": "膺窗穴的定位是什么?", "answer": "第3肋间隙，距前正中线4寸。"}\n{"question": "土源性线虫感染的多发地区是哪里？", "answer": "苏北地区；贵州省剑河县；西南贫困地区；桂东；江西省鄱阳湖区；江西省',
            'question": "膺窗穴的定位是什么?", "answer": "第3肋间隙，距前正中线4寸。"}\n{"question": "土源性线虫感染的多发地区是哪里',
            '背痛的放射治疗有些什么？", "answer": "单纯体外放射治疗"}\n{"question": "盆腔淤血症的症状',
            '应该是患阴道炎了，而且是细菌性阴道炎的可能性大，你可以用妇科洗剂擦洗外阴阴道，同时口服些妇科千金片或者花红颗粒，具体用药谨遵医嘱，如果效果不佳。较好到医院去做白带常规仔细检查，查明病因后必要时再出针对性的用药治疗。平常以清淡饮食为基础，尽量多喝水。白带增多患者可以在理解专业治疗的同时，在生活中用食疗做为辅助治疗，在日常生活饮食中需要有恰当饮食，防止刺激性食物。',
            '天雄的药用植物栽培"',
            '[3]\t "致病，如风寒、风热、风湿、风燥等。"}\n{"question": "ESD术的手术治疗有些什么？", "answer": "ESD术"}\n{"question": "为什么会得大脖子病", "answer": "甲状腺肿的病因还没完全清楚，情绪、药物、化学物质、放射线、遗传缺损、炎症、自身免疫等因素干扰甲状腺激素的合成、储存与释放，及血中存在刺激甲状腺生长的因子都可引起甲状腺肿。结节性甲状腺肿的CT特点为病灶多发、形态',
            '静脉曲张及腹壁静脉曲张等等。那么，静脉曲张的最好治疗方法都有哪些？下面为大家详细的介绍下相关的知识。1、中医中药疗法：这也是临床效果较为显著的静脉曲张治疗方法，使用疏通经络的药物改变血管内壁的弹性，快速修复受损的瓣膜，使凸起的团状、条索状血管团逐渐缩小、平滑，能从根本上调节身体的机能。2、下肢静脉曲张有穿弹力袜、注射硬化剂、手术剥除等治疗方法，深静脉瓣膜功能不全，可作瓣膜修复手术和腔镜下交通支结扎术等。下肢',
            '对患者有一定的影响。所以患者应该注意自身护理。流产患者注意自身护理，例如当的进行饮食调理，注意保暖避免着凉。',
            '膺窗穴的定位是什么?", "answer": "第3肋间隙，距前正中线4寸。"}\n{"question": "土源性线虫感染的多发地区是哪里？", "answer": "苏北地区；贵州省剑河县；西南贫困地区；桂东；江西省鄱阳湖区；江西省',
            'question": "梭形细胞RMS的多发群体是什么？", "answer": "儿童和成年人',
            '我想了解艾灸可以治疗宫颈糜烂吗？，我有轻度的宫颈糜烂，听人说可以用艾灸来治疗，艾灸可以治疗宫颈糜烂吗？", "answer": "宫颈糜烂这种疾病，决定以上这种物理治疗措施，可以起些配置治疗作用，但是不能替代药物',
            '盆腔淤血症是盆腔炎，有小腹痛，白带异常，尿频等症状。如有此病，你应该尽快去医院治疗，以免延误病情。平时做好妇科查体。治疗可以使用环丙沙星加甲硝唑。注意日常的卫生习惯，勤洗澡。盆腔患者应在平时生活中忌食白酒，白扁豆，鸭蛋，鸭肉；宜吃富含维生素的食物、富含有优质蛋白质的食物和膳食纤维类食物，如：鸡蛋，鸡肉，鸡心，鹌鹑蛋。忌吃腥发的食物，如：羊肉、鲢鱼、鲤鱼；忌吃不容易消化的食物，如：螺丝肉、豆芽；3、忌吃富含油脂的食物，如：肥肉、猪油、羊油。{"queson": "梭形细胞RMS的多发群体是什么？", "answer": "儿童和成年人"}{"question": "我想了解艾灸可以治疗宫颈糜烂吗？，我有轻度的宫颈糜烂，听人说可以用艾灸来治疗，艾灸可以治疗宫颈糜烂吗',
            '对患者有一定的影响。所以患者应该注意自身护理。流产患者注意自身护理，例如当的进行饮食调理，注意保暖避免着凉。"}\n{"question": "膺窗穴的定位是什么?", "answer": "第3肋间隙，距前正中线4寸。"}\n{"question": "土源性线虫感染的多发地区是哪里？", "answer": "苏北地区；贵州省剑河县；西南贫困地区；桂东；江西省鄱阳湖区；江西省',
        ]
        m.add_corpus(zh_list)
        q = ['梭形细胞RMS的多发群体是什么？', '膺窗穴的定位是什么?']
        for i in q:
            res = m.most_similar(i, topn=10)
            print('sim search: ', res)
            for c in res:
                print("search res:")
                print(f'\t{c}')
            print('-' * 50 + '\n')



if __name__ == '__main__':
    unittest.main()

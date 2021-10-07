#--*--usr/bin/env python --*--
#--*-- coding:utf-8 --*--
import os
import jieba
import gensim
currentPath = os.path.abspath('.').replace('\\', '/') + '/'
dataPath = 'dataset/'
dictPath = 'Dict/'
from SentimentAnalyzer import SentimentAnalyzer
from PreProcess import TrainXml2Pandas
from PreProcess import TestXml2Pandas

lstTestCase = [
    {'lineNum':14,'line':'章子怡是王家卫挑选出来的，在画面，音乐，剧情上靠得住的人才'},
    {'lineNum':15,'line':'她是电她是光'},
    {'lineNum':16,'line':'她是木叶村的希望，救世之主'},
    {'lineNum':17,'line':'内心的喜悦之情溢于言表，非常快乐快活'},
    {'lineNum':18,'line':'在剧情上起到了关键的作用，故事非常的好，有趣，有力'},
    {'lineNum':19,'line':'总体上来看就是有意思'},
    {'lineNum':20,'line':'有趣有趣，非常好玩'},
    {'lineNum':21,'line':'音乐上也不错'},
    {'lineNum':22,'line':'整体看来，好片！'},
    {'lineNum':23,'line':'开心 欢乐 有趣 乐观 很快 挺好'},
    {'lineNum':24,'line':'难过 难受 上心 悲哀 悲伤'},
    {'lineNum':25,'line':'不成 不行 拒绝 行不通 不可以 不接受 尚未 绝不 休想'}
]




class T2Analyzer:
    '''
        用于进行分析的分析器 这个类是之前编辑训练集分析器过程的解耦合
        主要是将标注的过程从分析器中剥离出来
        作为新的标注器来使用
        标注器使用的是词匹配的方式进行分析
        主分析器对传入的句子进行分词处理 通过查询句子中的词的出现情况 来对句子的描述对象进行分类
    '''
    modelPath = currentPath + 'movie.bin'
    setTypeT2 = {'uncertain', '人物', '其他', '剧情', '导演', '总体', '演员', '画面', '配乐'}

    pronWords = ['他', '她']
    #这个数据结构用于在进行有次序的判定过程中 首先进行哪个 本次分析的要求是先进行导演或者演员的判定
    #并且这个list内部的元素的顺序也代表着执行判定过程中的判定顺序
    setFirstClass = ['导演','演员']
    setSecondClass = ['人物', '总体', '剧情', '画面', '配乐']

    story = []
    music = []
    picture = []
    figure = []
    director = []
    actor = []

    dictType = {}

    w2v = gensim.models.Word2Vec

    #情感词典 分为正向词 负向词 否定词
    positive = []
    negative = []
    deny = []


    def __init__(self):
        print('开始数据初始化...')
        self.w2v = gensim.models.Word2Vec.load_word2vec_format(self.modelPath, binary=True)
        self.TypeWordsInit()
        self.TypeWordsExtend()
        self.SentimentWordsInit()
        print('数据初始化完成!')

    def TypeWordsInit(self):
        '''
        用于进行
            查询词词典的初始化 主要是从word2vec模型中筛选出前十个和标注分类相关的词组成集合
            导演和演员部分是读取一个导演 演员词典来进行构造的
        '''
        print('开始词表构建...')
        w2v = self.w2v
        self. story = [wordVec[0] for wordVec in w2v.most_similar('剧情', topn=11) if wordVec[0] != '但是']
        self.music = 'BGM bgm 好听 插曲 曲 歌舞 片尾曲 背景音乐 配乐 音效 配音 环绕 立体声 声 音乐 music Music'.split(' ')
        self.picture = [wordVec[0] for wordVec in w2v.most_similar('画面', topn=10)]
        self.figure = [wordVec[0] for wordVec in w2v.most_similar('人物', topn=10)]
        self.director = [line.replace('\n', '') for line in open(currentPath + 'lstDirector', encoding='utf-8')]
        self.actor = [line.replace('\n', '') for line in open(currentPath + 'lstActor', encoding='utf-8')]
        self.SetDictType()
        print('词表构建完成!')

    def TypeWordsExtend(self):
        '''
        对原有的词典进行扩充的函数
        功能在于对直接从训练模型中以及词典中读取的不足的地方进行扩展
        目前还只是

        '''
        print('开始词表扩展...')
        storyExtra = '剧情 隐喻 推进 噱头 台词 吐槽 槽点 俗套 背景 铺垫 真实 ' \
                     '开局 终局 煽情 励志 青春 戏剧性 字幕 线索 前段 后段 片尾 结局'.split(' ')
        # musicExtra = '配音 音效 曲 配乐 好听 歌舞 音乐'.split(' ')
        pictureExtra = '视觉 布景 场面调度 镜头 质地 画面 场面 场景 布景'.split(' ')
        self.story += storyExtra
        # self.music += musicExtra
        self.picture += pictureExtra
        self.SetDictType()
        print('词表扩展完成!')

    def SetDictType(self):
        '''
        这个数据结构在分析过程中至关重要
         分析方法需要通过类型进行词表的查找
         这种对应关系就存储在这里
        '''
        print('分析字典重构开始...')
        self.dictType = {
            '剧情': self.story,
            '配乐': self.music,
            '画面': self.picture,
            '人物': self.figure,
            '总体': [],
            '导演': self.director,
            '演员': self.actor
        }
        print('分析字典重构完成!')

    def SentimentWordsInit(self):
        '''
        用于对分析器中的情感词典进行初始化
         情感词典分为正向负向 否定词
        '''
        print('情感词典构建开始...')
        self.deny = [word.replace('\n', '') for word in
                     open(currentPath + dictPath + 'deny.csv', 'r', encoding='utf-8').readlines()]

        self.positive = [word.replace('\n', '') for word in
                        open(currentPath + dictPath + 'positive.csv', 'r', encoding='utf-8').readlines()]

        self.negative = [word.replace('\n', '') for word in
                        open(currentPath + dictPath + 'negative.csv', 'r', encoding='utf-8').readlines()]

        print('情感词典构建完成!')



    '''下面是原始的分析部分 解耦合之前的分析器的内容
    这部分的基本策略主要有以下几点：
    1.基础的 通过 查询句子中的词是否在词典中出现 以这为依据进行分类 比如句中出现了导演的名字 则直接判定为描述导演的句子
    2.判定是分为两个阶段的 首先判定是否为导演或者演员 若在此步骤中完成判定 则本句子的性质就由此确定了
        若导演和演员名称均未出现 则执行下一阶段的判定
    3.画面 音乐 剧情 人物 分别用word2Vector 进行词扩展 扩展得到的词作为词表 在这一阶段的分析中
        若句子中的词出现在了 上述类别中 则判定为这一类
    4.通过设置flag来标志分类是否完成 上述步骤中均未捕获的句子 统一分类作为 总体

    Extra 规则 为了进行精度的提高 引入以下几个规则来对 Train集合中的特性进行拟合
    1.在第一个Text块中 发现 事实上每个句子都是同一个块分句之后形成的
        后续句子中的一些代词可能是对第一个句子中的某个演员/导演/角色 的代称 因此 引入规则：
        在判定出一句话为导演/演员 时 将整个（或者后续）句子块中的所有代词替换为这个 演员/导演的名字 再执行判定


    2.首先对描述剧情的词组的库进行扩展 以满足准确率的要求
    通过数据分析发现 对于剧情的判定往往带有一种 连坐的形式
        Train数据集中 同一个ID下的句子可以被认为是一个句子组
        同一组句子若出现了对于剧情的判定 由于语言的逻辑性 往往不能一句话表明
        这使得一组句子均为描述剧情的可能性非常之高 因此 引入规则：
        若同一个ID之下（在Train中表现为1-100） 的句子中出现了 剧情 这一分类的判定
        则将这一句子组中的 判定为 总体（未判定出属性） 的句子 改判为 剧情

    '''

    '''
        重构代码之后 将分析器部分从原有的类中独立出来
        其间发生了架构的改变 因此之前的一些策略需要有所改变
        1.基本策略中 有一个判定次序的问题
        即先进行导演 演员的判定 之后执行 具体类别的判定 因此 这种判定顺序应该在这个分析器中继承下来
        2.对未捕获行为的句子 最终应当执行策略 判定其为总体

        3.极为重要的一点在于 之前的判定过程中 对于单个句子的判定 一定程度上依赖了整个句子篇章的一些特性
            训练集中的数据给出时是以 <ID val=1>句子</ID>的形式给出的
            每个ID之下的句子都可以认为是一个篇章之内的句子 因此通过对其整体行为的分析 往往有奇效
            而分析器是一个独立的类 并不能再次和文本处理过程深度耦合 因此需要进行再次设计以保证原有的分析思想的保证
            因此首先考虑的就是设计两个不同层级的接口
            一个接口可以接受一个句子作为参数 将这个句子分词之后逐个进行分析 给出这个句子的类别
            另一个接口接受一组句子作为参数 这些句子被默认为处于一个篇章内部 通过篇章内部的关系来判定句子整体性的分类
        4.原来的代词预处理策略以及连坐判定剧情的方法可以通过这种传入一组句子的方法来实现
    '''

    def SingleLineTypeJudge(self, line):
        '''
            传入参数为一个句子 line为str类型的参数
            函数处理流程如下
            首先对line进行分词操作 之后进行导演--演员的判别 返回一个dict
            dict格式如下

            {'type':判定的类别信息,'keyWord':关键词信息}

            这种返回格式是为了后续的处理的方便
            例如代词替换过程中 需要了解当前判定出的演员或者导演的名字
            在后续处理过程中需要剧情的信息 等
            这是原来的ClassifyByClassicMethod 方法的另一个版本
        '''
        #初始化这两个值 以便在分析过程中进行记录
        type = 'badass'
        keyWord = 'motherfucker'
        if line == None:
            line = '无'
        #将句子分词形成一个词表 尾巴后面加一个空格 防止词表为空所导致的种种问题
        words = [word for word in jieba.cut(line)] + [' ']
        #执行判定的过程 先进行导演或者演员的判定 之后进行类型的详细判定
        for word in words:
            #先进行导演和演员的判定
                for firstType in self.setFirstClass:
                    if type == 'badass' and keyWord == 'motherfucker':
                        #当单词在这里面的时候 写记录
                        if word in self.dictType[firstType]:
                            type = firstType
                            keyWord = word

        #当判定结果表明 上面的过程中没有找出来导演或者演员的名字的时候 执行下面的分类过程
        if type == 'badass' and keyWord == 'motherfucker':
                #对第二顺位的类型执行判断
                for word in words:
                    if type == 'badass' and keyWord == 'motherfucker':
                        for secondType in self.setSecondClass:
                            if word in self.dictType[secondType]:
                                type = secondType
                                keyWord = word

        #通过两个阶段的分析过后 还是没有区分出结果 则将其类型结果写为 总体 关键词也写为 总体
        if type == 'badass' and keyWord == 'motherfucker':
            type = '总体'
            keyWord = '总体'
        return {'type': type, 'keyWord': keyWord}

    def ChatperJudge(self, lstLineInDict):
        '''
        篇章级的分析 输入量是一个包含着dict的list对象
        其中元素的定义如下

        {'lineNum':句子的行号:, 'line': 句子本体}

        这种写法的目的在于 便于在之后的结果更新中更为方便 实际上 元组本身的第一个元素并不重要
        在传入之后会原样传出

        返回值为一个 三元组list 即在传入的参数中元组的后面 追加一个属性 该属性为这个句子的最终判定类型
        在此处 为了便于分析 将三元组用list的形式来表示
        [{'lineNum':句子的行号:, 'line': 句子本体,'judge': 判定得到的类型}
        judge 条目中存储的是 {'type':判定的类别信息,'keyWord':关键词信息}
        '''
        lstOutput = []
        #首先调用SingleLineJudge 进行单个句子的类型信息判断 判断之后在字典对象中最佳一个judge属性 之后加入到lstOutput中
        for elemDict in lstLineInDict:
            elemDict['judge'] = self.SingleLineTypeJudge(elemDict['line'])
            lstOutput.append(elemDict)


        #执行原始的策略一 将代词替换为曾经出现的演员的名字或者导演名字
        keyWord = 'buster'
        for item in lstOutput:
            if item['judge']['type'] == '导演' or item['judge']['type'] == '演员':
                keyWord = item['judge']['keyWord']

        #当判定句子组中包含着导演或者演员时 执行代词替换的过程 将句子中的所有代词换为 导演或者演员的名字
        if keyWord != 'buster':
            for lineNum,item in enumerate(lstOutput):
                for pronWord in self.pronWords:
                    line2Replace = lstOutput[lineNum]['line']
                    lstOutput[lineNum]['line'] = line2Replace.replace(pronWord, keyWord)

        #执行完毕句子中的代词替换的工作之后 对其再进行一次分析操作 得到新的值 就是最终值了

        for lineNum,elemDict in enumerate(lstOutput):
            lstOutput[lineNum]['judge'] = self.SingleLineTypeJudge(elemDict['line'])


        #执行完毕代词替换策略之后 执行连坐测试的策略 即原程序代码中的策略二
        #即 若组中出现剧情的判定 将同一篇章中的所有判定为总体的 改判为剧情
        storyFlag = 0
        for item in lstOutput:
            if item['judge']['type'] == '剧情':
                storyFlag = 1
        if storyFlag == 1:
            for lineNum,item in enumerate(lstOutput):
                if item['judge']['type'] == '总体':
                    lstOutput[lineNum]['judge']['type'] = '剧情'
        #判断完毕 返回dict
        return lstOutput

    def SingleLineSentimentJudge(self, line):
        '''
        传入一个句子 执行句子级的情感判定操作
        执行的策略如下
        首先进行分词 读取每个词
        查看每个词是否在情感词典中 有则将其（多个就都记录）记录下来
        当存在情感词时 执行情感分析器 判断正向或者负向情感倾向 给出标注
        当不存在情感词时 判定为uncertain
        额外规则 当分词之后出现某些词为转折词的话 判定其为感慨选项 返回 感慨词（感慨词可以通过训练集进行过拟合）

        额外进行感慨词的判定 当判定的情感信息为中性 且出现了否定词 则将其归类为 感慨 词语为否定词

        返回的格式如下
        {'sentiment': 判定得到的情感值,'word':判据}
        '''
        if line == None:
            return {'sentiment':'uncertain','word':'None'}
        words = [word for word in jieba.cut(line)]
        analyzer = SentimentAnalyzer()
        analyzer.load()

        # wordsPositive = []
        # wordsNegative = []
        wordsDeny = []

        # 根据任务一做出来的结果 用这个情感分析器来评定情感值时 若结果落在[-0.35,0.35]之间的话 则判定为中性

        flagSentiment = 0

        for word in words:
        #     if word in self.positive:
        #         wordsPositive.append(word)
        #     if word in self.negative:
        #         wordsNegative.append(word)
            if word in self.deny:
                wordsDeny.append(word)

        #情感分析器阶段
        valAnalyzer,wordsAll = analyzer.analyze(line)
        if valAnalyzer < -0.35:
            flagSentiment = -1
        elif valAnalyzer > 0.35:
            flagSentiment = 1
        elif valAnalyzer >= -0.35 and valAnalyzer <= 0.35:
            flagSentiment = 0
        # allIn = wordsDeny + wordsPositive + wordsNegative
        #全部的判定流程


        # if len(wordsDeny) != 0 or len(wordsNegative) != 0 or len(wordsPositive) != 0:
        if len(wordsAll) != 0:
            if flagSentiment == 0 and len(wordsDeny) != 0:
                return {'sentiment': '感慨', 'word': ','.join(wordsAll)}
            return {'sentiment': str(flagSentiment), 'word': ','.join(wordsAll)}
        else:
            return {'sentiment': 0, 'word': 'None'}


        #
        # if flagSentiment == 1:
        #     if len(wordsPositive) == 0:
        #         return {'sentiment':str(flagSentiment),'word':'None'}
        #     else:
        #         return {'sentiment': str(flagSentiment), 'word': ','.join(allIn)}
        # elif flagSentiment == -1:
        #     if len(wordsPositive) == 0:
        #         return {'sentiment':str(flagSentiment),'word':'None'}
        #     else:
        #         return {'sentiment': str(flagSentiment), 'word': ','.join(allIn)}
        # elif flagSentiment == 0:
        #     if len(wordsDeny) != 0:
        #         return {'sentiment': '感慨', 'word': ','.join(wordsDeny)}
        #     elif len(wordsDeny)!= 0 or len(wordsNegative) != 0 or len(wordsPositive) != 0:
        #         return {'sentiment':str(flagSentiment), 'word':','.join(allIn)}
        #     else:
        #         return {'sentiment': str(flagSentiment), 'word': 'None'}

    def TextSentimentJudge(self,textDict):
        '''
        传入参数的形式是一个dict的形式 为了处理的方便
        {'id':text块的id,'text':text主体内容}

        return 的值也是一个相同的dict 内容是对其情感极性的判定
        {'id':具体是哪一块,'sentiment':情感值 1 0 或者 -1}
        '''
        analyzer = SentimentAnalyzer()
        analyzer.load()
        sentiVal = analyzer.analyze(textDict['text'])
        flagSentiment = 0
        if sentiVal < -0.35:
            flagSentiment = -1
        elif sentiVal > 0.35:
            flagSentiment = 1
        else:
            flagSentiment = 0

        return {'id':textDict['id'],'sentiment':flagSentiment}




# pop = T2Analyzer()
# def testCase():
#     lst = []
#     lst.append({'lineNum': 10, 'line': '剧情上音乐上合适的人章子怡是王家卫选出来的'})
#     lst.append({'lineNum': 11, 'line': '她是天选之子'})
#     lst.append({'lineNum': 12, 'line': '她是木叶村的希望'})
#     lst.append({'lineNum': 13, 'line': '她是motherfucker'})
#     lst.append({'lineNum': 14, 'line': '总体上来看剧情是不错的'})
#     lst.append({'lineNum': 15, 'line': '总体'})
#     lst.append({'lineNum': 16, 'line': '总体'})
#     return lst

# lstt = testCase()
# result = pop.ChatperJudge(lstt)
# # rst = pop.SingleLineJudge('总体上来看剧情是不错的')
# for i in result:
#     print(i)


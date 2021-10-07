#-*- coding:utf-8 -*-
import jieba
import math
class SentimentAnalyzer(object):
    """description of class"""
    def __init__(self):
        self.posi_type = set(['PA','PE','PD','PH','PG','PB','PK'])
        self.load()
        pass
    
    def map_sentiment(self,type):
        if type in self.posi_type:
            return 1
        else:
            return 2

    def load(self):
        self.level_dict = {}
        level_match = [0,0.5,0.5,1.0,1.0,1.5,1.5,1.75,1.75,2.0,2.0]
        sen_match = [0,1.0,1.0,1.25,1.25,1.5,1.5,1.75,1.75,2.0,2.0]
        #为程度词映射权重，存储在level_dict中
        for line in open(u'Dict\程度级别词语.csv',encoding='utf-8').readlines()[1:]:
            item = line.rstrip('\n').split(',')
            self.level_dict[item[0]] = level_match[int(item[1])]
        #读取否定词
        self.deny_set = set([item.rstrip('\n') for item in open(u'Dict\deny.csv',encoding='utf-8').readlines()[1:]])
        #读取情感词表
        self.sen_dict = {}   #得到的二元组，首先是极性，然后是强度
        for line in open(u'Dict\positive.csv',encoding='utf-8').readlines():
            item = line.rstrip('\n')
            self.sen_dict[item] = (1,1)
        for line in open(u'Dict/negative.csv',encoding='utf-8').readlines():
            item = line.rstrip('\n')
            self.sen_dict[item] = (2,1)
        for line in open(u'Dict\情感词汇本体.csv',encoding='utf-8').readlines()[1:]:
            item = line.rstrip('\n').split(',')
            self.sen_dict[item[0]] = (self.map_sentiment(item[4]),sen_match[int(item[5])])

    def analyze(self,content):
        sen_q = []
        keywords=[]
        terms = list(jieba.cut(content))
        q_total = 0.0
        num_senti = 1e-6
        for i,term in enumerate(terms):
            if term in self.sen_dict:
                keywords.append(term)
                num_senti += 1
                posi_coef = self.sen_dict[term][0]  #情感极性
                if posi_coef == 2:
                    posi_coef = -1.0
                elif posi_coef == 1:
                    posi_coef = 1.0
                else:
                    continue
                sen_q.append((i,self.sen_dict[term][0],self.sen_dict[term][1],term))  #情感词四元组（位置，情感极性，强度，词）
                #进行情感强度计算
                val = 1.0 * posi_coef * self.sen_dict[term][1]  #计算情感词强度
                deny_coef = -1.0  #否定词的系数
                for i in range(len(sen_q) - 2,-1,-1):
                    if len(sen_q[i]) == 4:  #如果是四元组，则是上一个情感词，则退出循环
                        break
                    if sen_q[i][1] > 0: #遇上程度词
                        val = val * sen_q[i][1]
                        deny_coef = 0.5   #如果程度词在否定词后，如“很不好”，则否定词权重变为0.5，否则为-1
                    elif sen_q[i][1] == -1:   #遇上否定词
                        val = val * deny_coef  
                        deny_coef = -1.0
                q_total+=val
            elif terms[i] in self.deny_set:
                sen_q.append((i,-1,term))    #否定词三元组（位置，-1，词）
            elif terms[i] in self.level_dict:
                sen_q.append((i,self.level_dict[term],term)) #修饰强度三元组（位置，强度，词）
        q_ava = scal_regression(q_total,num_senti)
        return q_ava,keywords

def scal_regression(num,total):
    if total==1:
        return 0.25

    if num ==4 and total == 2:
        return 0.40
    if num<0:
        q_ava = - math.sqrt(math.fabs(num))/total
    else:
        q_ava = math.sqrt(math.fabs(num))/total
    return q_ava

if __name__=="__main__":
    analyzer=SentimentAnalyzer()
    analyzer.load()
    sentence = '今儿是个好天气！'
    print(sentence)
    res = analyzer.analyze(sentence)
    print(res)
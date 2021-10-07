#--*--usr/bin/env python --*--
#--*-- coding:utf-8 --*--
'''
对解耦合之后的算法进行验证 验证结果表明和原来的性能一致
解耦合成功

'''



from T2Analyzer import T2Analyzer as analyzer
from PreProcess import TrainXml2Pandas
import os
currentPath = os.path.abspath('.').replace('\\', '/') + '/'
dataPath = 'dataset/'

marker = analyzer()

Train = TrainXml2Pandas(currentPath + dataPath + 'train_data.xml')
Train = Train.reset_index()
Train = Train.drop('index', axis=1)
Train = Train.reset_index()

lstTrain = []

for key in range(0,len(Train)):
    dictTemp = {}
    dictTemp['lineNum'] = Train.index[key]
    if Train.text[key] == None:
        dictTemp['line'] = '无'
    else:
        dictTemp['line'] = Train.text[key]
    dictTemp['id'] = Train.id[key]
    lstTrain.append(dictTemp)

lstReshape = []
for ide in range(0,100):
    ident = ide + 1
    lstTemp = []
    for item in lstTrain:
        dictTemp = {}
        if item['id'] == str(ident):
            dictTemp['linNum'] = item['lineNum']
            dictTemp['line'] = item['line']
            lstTemp.append(dictTemp)
    lstReshape.append(lstTemp)

lstAnalyzed = []
for lstLineInDict in lstReshape:
    lstAnalyzed +=marker.ChatperJudge(lstLineInDict)


lstJudge = []
for key in range(0,len(Train)):
    lstJudge.append({'lineNum':Train.index[key],'t2':Train.t2[key]})


dictP = {}
dictR = {}
dictF = {}

dictTypeNumInT2 = {}
dictTypeNumInAnal = {}
dictRight = {}

setTypeT2 = {'uncertain', '人物', '其他', '剧情', '导演', '总体', '演员', '画面', '配乐'}
setTypeJudge = {'人物', '其他', '剧情', '导演', '演员', '画面', '配乐'}

# 统计数字时顺便将其初始化 原来样本中有都少个对应的值
for keys in setTypeT2:
    dictRight[keys] = 0
    tmpCounter = 0
    for item in lstJudge:
        if keys == item['t2']:
            tmpCounter += 1
    dictTypeNumInT2[keys] = tmpCounter

# 统计分析过后 各个类中都分析出了几个 用于计算查准率 我们分析出来了多少个对应的值
for keys in setTypeT2:
    tmpCounter = 0
    for item in lstAnalyzed:
        if item['judge']['type'] == keys:
            tmpCounter += 1
    dictTypeNumInAnal[keys] = tmpCounter

# 查对的数目一共有多少个 在这里计算
for lineNum in range(0, len(lstJudge)):
    typeT2 = lstJudge[lineNum]['t2']
    typeAnal = lstAnalyzed[lineNum]['judge']['type']
    if typeT2 == typeAnal:
        dictRight[typeT2] += 1

for keys in setTypeT2:
    dictR[keys] = dictRight[keys] / dictTypeNumInT2[keys]

for keys in setTypeT2:
    if keys == 'uncertain' or keys == '其他':
        dictP[keys] = 0
    else:
        dictP[keys] = dictRight[keys] / dictTypeNumInAnal[keys]

for keys in setTypeT2:
    if dictR[keys] == 0 or dictP[keys] == 0:
        dictF[keys] = 0
    else:
        dictF[keys] = 2 / (1 / dictP[keys] + 1 / dictR[keys])



import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=os.path.expandvars(r"%windir%\fonts\simsun.ttc"), size=14)

def DrawBarByDict(dictObj):
    '''
    dictObj 是一个字典 键只对应一个float值
    '''
    x = [i for i in dictObj.keys()]
    x1 = [i+1 for i in range(0,len(x))]
    y = [dictObj[j] for j in x]
    plt.bar(x1, y,width = 0.35,facecolor = 'lightskyblue',edgecolor = 'white')
    plt.xticks([i + 0.35/2 for i in x1] ,x,fontproperties=font)
    plt.show()

DrawBarByDict(dictR)
DrawBarByDict(dictP)
DrawBarByDict(dictF)


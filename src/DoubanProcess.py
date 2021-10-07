#--*--usr/bin/env python --*--
#--*-- coding:utf-8 --*--
from T2Analyzer import T2Analyzer as analyzer
import os
currentPath = os.path.abspath('.').replace('\\', '/') + '/'
dataPath = 'dataset/'
import re
import pickle

#用于进行句子分割的标点符号pattern集合
puncutation = ', \. ; \? \! \* ， 。 ？ ！'.replace(' ','|')
patternPuncutation = re.compile(puncutation)
#用于进行字符串分割测试的样例

lineTestCase = '我爱北京天安门，天安门上太阳升。伟大领袖毛主席,指引我们向前进.啦啦啦!lalala！\
每个人脸上都笑开颜？娃哈哈娃哈哈?每个人脸上都笑开颜*yes!'

dictDouBanText = {}
with open(currentPath + 'comments_label.csv', 'r', encoding='utf-8') as file:
    for lineNum,line in enumerate(file.readlines()):
        dictDouBanText[lineNum] = re.split(patternPuncutation,(line.split(',') + ['无'])[1].replace('\n',' '))

marker = analyzer()

dictResult = {}
count = 0
print(len(dictDouBanText.keys()))


for key in dictDouBanText.keys():
    print('Preparing analyze line:%d...' % key)
    lines = dictDouBanText[key]
    lstDictInLine = []
    for line in lines:
        dictTemp = {}
        dictTemp['lineNum'] = key
        dictTemp['line'] = line
        lstDictInLine.append(dictTemp)
    dictResult[key] = marker.ChatperJudge(lstDictInLine)

import pickle

with open('DictDoubanMarked.pkl', 'wb') as file:
    pickle.dump(dictResult, file=file)

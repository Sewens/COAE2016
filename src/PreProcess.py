#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import pandas as pd



currentPath = os.path.abspath('.').replace('\\', '/') + '/'
dataPath = 'dataset/'
dictPath = 'Dict/'




def TrainXml2Pandas():
    with open(currentPath + dataPath + 'train_data.xml', 'r', encoding='utf-8') as file:
        trainXmlTree = ET.ElementTree(file=file)
    treeRoot = trainXmlTree.getroot()
    Train = pd.DataFrame(data=[], columns=['id', 't1', 't2', 't3_sentiment', 't3_word', 'text'])
    elemsDoc = treeRoot.findall('DOC')
    for elemDoc in elemsDoc:
        idElem = elemDoc.find('ID')
        # 获取到ID的值
        idVal = idElem.text

        sentenceElems = elemDoc.find('TEXT').findall('SENTENCE')
        # 获取到每个句子的情况
        lstText = [node.text for node in sentenceElems]

        taskElems = elemDoc.find('REFERENCE').findall('TASK')

        # 对T1子任务的处理
        task1Val = taskElems[0].text
        # 对T2子任务的处理
        task2Lst = [node.text for node in taskElems[1]]
        # 对T3子任务的处理
        task3Lst = [(node.find('SENTIMENT').text, node.find('WORD').text) for node in taskElems[2]]
        dictTmp = {}
        dictTmp['id'] = idVal
        dictTmp['text'] = lstText
        dictTmp['t1'] = task1Val
        dictTmp['t2'] = task2Lst
        dictTmp['t3_sentiment'] = [item[0] for item in task3Lst]
        dictTmp['t3_word'] = [item[1] for item in task3Lst]
        pdTmp = pd.DataFrame(dictTmp)
        Train = pd.concat([Train, pdTmp])
    Train = Train.reset_index()
    Train = Train.drop('index', axis=1)
    return Train

def TestXml2Pandas():
    with open(currentPath + dataPath + 'test_data.xml', 'r', encoding='utf-8') as file:
        testXmlTree = ET.ElementTree(file=file)
    treeRoot = testXmlTree.getroot()
    Test = pd.DataFrame([], columns=['id', 'text'])
    elemsDoc = treeRoot.findall('DOC')
    for elemDoc in elemsDoc:
        dictTmp = {}
        idElem = elemDoc.find('ID')
        idVal = idElem.text

        textElem = elemDoc.find('TEXT')
        textLst = [item.text for item in textElem]

        dictTmp['id'] = idVal
        if len(textLst) == 0:
            textLst = ['无']
        dictTmp['text'] = textLst

        dpTmp = pd.DataFrame(dictTmp)
        Test = pd.concat([Test, dpTmp])
    Test = Test.reset_index()
    Test = Test.drop('index', axis=1)
    return Test





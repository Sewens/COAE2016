#--*--usr/bin/env python --*--
#--*-- coding:utf-8 --*--
from T2Analyzer import T2Analyzer
import os
import pandas as pd
from PreProcess import TestXml2Pandas
import xml.etree.ElementTree as ET

currentPath = os.path.abspath('.').replace('\\', '/') + '/'
dataPath = 'dataset/'
dictPath = 'Dict/'

marker = T2Analyzer()

Test = TestXml2Pandas()

#此处写子任务一的所有代码
lstTask1Val = []
for ident in sorted(set([int(i) for i in Test.id])):
    text = ''
    for line in Test[Test.id == str(ident)].text:
        text += line
    lstTask1Val.append(marker.TextSentimentJudge({'id':ident,'text':text}))

''' 这部分代码是将结果依原样写成xml的代码'''
# root = ET.Element('DATA')
# for counter, item in enumerate(lstTask1Val):
#
#     subDoc = ET.SubElement(root, 'DOC')
#     subID = ET.SubElement(subDoc, 'ID')
#     subID.text = str(item['id'])
#     subREF = ET.SubElement(subDoc, 'REFERENCE')
#     subTask = ET.SubElement(subREF, 'TASK')
#     subTask.set('ID', '1')
#     subTask.text = str(item['sentiment'])
#
#     tree = ET.ElementTree(root)
#
#     tree.write('COAE_task2_1_0.xml', method='xml',encoding='utf-8')

title = '2-1\t'
org = 'dutir\t'
idLine = '0000'
with open('COAE_task2_1_1.txt', 'w', encoding='utf-8') as file:
    for count,dictTmp in enumerate(lstTask1Val):
        idLine = str(10000 + int(Test.id[count]))
        line = title + org + idLine + '\t' + str(dictTmp['sentiment']) + '\n'
        file.write(line)



#此处是子任务二所有代码
lstTask2Val = []
for ident in sorted(set([int(i) for i in Test.id])):
    lstDictText = []
    for line in Test[Test.id == str(ident)].index:
        tmpDict = {}
        tmpDict['lineNum'] = line
        tmpDict['line'] = Test.text[line]
        lstDictText.append(tmpDict)
    lstTask2Val.append(marker.ChatperJudge(lstDictText))

'''这部分代码是将结果依原样写成xml的代码'''
# root = ET.Element('DATA')
# for counter, item in enumerate(lstTask2Val):
#
#     subDoc = ET.SubElement(root, 'DOC')
#     subID = ET.SubElement(subDoc, 'ID')
#     subID.text = str(counter)
#     subREF = ET.SubElement(subDoc, 'REFERENCE')
#     subTask = ET.SubElement(subREF, 'TASK')
#     subTask.set('ID', '2')
#     for lineNum, dictTmp in enumerate(item):
#         subSentence = ET.SubElement(subTask, 'SENTENCE')
#         subSentence.set('ID', str(lineNum))
#         subSentence.text = dictTmp['judge']['type']
#
#     tree = ET.ElementTree(root)
#     tree.write('COAE_task2_2_0.xml', method='xml', encoding='utf-8')

title = '2-2\t'
org = 'dutir\t'
idLine = '0000'
with open('COAE_task2_2_1.txt', 'w', encoding='utf-8') as file:
    for count,lstDict in enumerate(lstTask2Val):
        ident = count + 1
        idLine = str(10000 + ident)[1:]
        for lineNum,item in enumerate(lstDict):
            subIdLine = idLine+ '_' + str(10 + lineNum + 1)[1:] + '\t'
            typeStr = item['judge']['type']
            line = title + org + idLine + '\t' + subIdLine + typeStr + '\n'
            file.write(line)


#此处是子任务三的所有代码
lstTask3Val = []
for ident in sorted(set([int(i) for i in Test.id])):
    lstDictText = []
    for line in Test[Test.id == str(ident)].index:
        tmpDict = {}
        tmpDict['lineNum'] = line
        tmpDict['line'] = marker.SingleLineSentimentJudge(Test.text[line])
        lstTask3Val.append(tmpDict)

for lineNum in range(0,len(Test)):
    lstTask3Val[lineNum]['id'] = Test.id[lineNum]

title = '2-3\t'
org = 'dutir\t'
idLine = '0000'
lstTp = [item.split('\t')[4].replace('\n','') for item in open('COAE_task2_3_1.txt', 'r', encoding='utf-8').readlines()]
with open('COAE_task2_3_1.txt', 'w', encoding='utf-8') as file:
    for count in range(0, 400):
        ident = count + 1
        idLine = str(10000 + ident)[1:]

        for currentLine, item in enumerate(lstTask3Val):
            counter = 0
            if item['id'] == str(count):
                subIdLine = idLine + '_' + str(10 + counter + 1)[1:]
                counter += 1
                typeStr = lstTp[currentLine]
                sentimentVal = str(item['line']['sentiment'])
                if item['line']['sentiment'] == None:
                    item['line']['sentiment'] = 'None'
                words = '\t'.join(item['line']['word'].split(','))
                line = title + org + idLine + '\t' + subIdLine + '\t' + typeStr + '\t' + sentimentVal + '\t' + words + '\n'
                file.write(line)

# root = ET.Element('DATA')
# for identify in sorted(set([i for i in Test.id])):
#
#     subDoc = ET.SubElement(root, 'DOC')
#     subID = ET.SubElement(subDoc, 'ID')
#     subID.text = str(identify)
#     subREF = ET.SubElement(subDoc, 'REFERENCE')
#     subTask = ET.SubElement(subREF, 'TASK')
#     subTask.set('ID', '2')
#
#     for count, dictTmp in enumerate([item for item in lstTask3Val if item['id'] == str(identify)]):
#         count += 1
#         subSentence = ET.SubElement(subTask, 'SENTENCE')
#         subSentence.set('ID', str(count))
#         subSentiment = ET.SubElement(subSentence, 'SENTIMENT')
#         subSentiment.text = dictTmp['line']['sentiment']
#         subWord = ET.SubElement(subSentence, 'WORD')
#         subWord.text = dictTmp['line']['word']
#
#     tree = ET.ElementTree(root)
#     tree.write('COAE_task2_3_0.xml', method='xml', encoding='utf-8')


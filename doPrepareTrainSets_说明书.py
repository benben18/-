# -*- coding: utf8 -*-

import pandas as pd
import numpy as np

symPath='./prepareTrainSets/Indication.csv'
# disPath='datasets/prepareTrainSets/diseaseMatch.csv'
disDictName = "prepareTrainSets/disease_new2.dic"
symDictName = "prepareTrainSets/symptom_new2.dic"
bodyDictName = "prepareTrainSets/body中文身体部位名称.dic"


def loadDict(dicName, inType):
    bodyDict = dict()
    for item in open(dicName,encoding='utf8'):
        bodyDict[item.strip().replace('\n','')] = inType
    return bodyDict

###### 构建词典
#疾病词典包括互联网爬取的疾病名称、疾病别名、ICD10疾病名称，去重后共39615条数据
disDict = loadDict(disDictName, 'DISEASE')

#症状为互联网爬取的症状描述，去重后共7457条数据；
symDict = loadDict(symDictName, 'SYMPTOM')

#身体部位为互联网爬取的身体部位描述，去重后共1929条数据。
bodyDict = loadDict(bodyDictName, 'BODY')



# 加载待处理的文本,对规范化的句子，使用词典中的每个词进行全匹配，记录匹配的词、词的起始index、词的结束index和实体类型。
'''row 为规范化的句子
name 从字典中查询句子中存在的疾病名称
typename 为匹配的字典名称'''
def row2ner(result, row, name,typeName):
    p = row.find(name,0)
    while(p!=-1):
        result.append(name+' '+str(p)+' '+str(p+len(name))+' '+typeName)
        p = row.find(name, p+1)
    # print(result)

'''
将检测出的实体转化成BIO格式
'''
def ner2lable(bio, des,inResult, btype ,itype):
    for i in range(len(inResult)):
        inStr = inResult[i]
        #inArr = inStr.split(" ")
        s = int(inStr[1])
        e = int(inStr[2])
        bio[s] = btype+"-"+inStr[3]
        for j in range(s+1, e):
            bio[j] = itype+'-'+inStr[3]

#提取症状
columnName = "适应症"
#保存的train数据
trainPath = "prepareTrainSets/ner_train_data.txt"

# loadDiseaseDatasets(symPath, columnName, trainPath)
# loadDiseaseDatasets(disPath, columnName, trainPathDis)

df_dis = pd.read_csv(symPath)


df_dis = df_dis.dropna()
desList = df_dis[columnName].tolist()#症状
f = open(trainPath, "w", encoding='utf8')
print(len(desList))
for i in range(len(desList)):
    print('reading the '+ str(i) +' data from deslist')
    des = desList[i]  #读取症状的内容
    # for des in desList[0:100]:
    result = []
    # print("des:", len(des),des)
    if not des:
        continue
    # des格式化， bio初始化为O
    des = des.replace(' ', '').replace('\t', '').replace('\n', '').replace('　', '').strip() #去除换行符
    # print(des)
    bio = ['O' for i in range(len(des))]
    # 检索所有的疾病，记录起始位置
    typeName = 'DISEASE'
    for dis in disDict:
        row2ner(result, des, dis, typeName)
    # print(result)

    # ner2lable(bio, des, result, 'B-DIS','I-DIS')
    # 检索所有的症状，记录起始位置
    result1 = []
    typeName = 'SYMPTOM'
    for sym in symDict:
        row2ner(result1, des, sym, typeName)
    # print(result1)
    # ner2lable(bio, des, result1, 'B-SYM','I-SYM')
    # 检索所有的身体部位，记录起始位置
    result2 = []
    typeName = 'BODY'
    for body in bodyDict:
        row2ner(result2, des, body, typeName)
    # print(result2)
    # ner2lable(bio, des, result2, 'B-BODY','I-BODY')
    # print(len(bio),bio)

    result4 = result + result1 + result2
    # print("result4=",len(result4),result4)
    # 字符串转二维数组
    result5 = [[0 for i in range(5)] for j in range(len(result4))]
    for i in range(len(result4)):
        resArr = result4[i].split(' ')
        result5[i][0] = resArr[0]
        result5[i][1] = int(resArr[1])
        result5[i][2] = int(resArr[2])
        result5[i][3] = resArr[3]
        result5[i][4] = len(resArr[0])
    # idex=np.lexsort([result4[:,1]])
    # sorted_data = index[idex, :]
    # 按照起始位置和实体长度排序
    result5.sort(key=lambda x: (x[1], x[4]))
    # print("5=",result5)
    # 选择实体词最长的进行最大匹配
    result6 = [[0 for i in range(5)] for j in range(len(result5))]
    maxIndexNum = 0
    maxIndexAll = 0
    i = 0
    # print("len 5 =", len(result5))
    # 迭代检索实体词，如果后面的实体词和当前实体词起始索引一致，则找最长的实体，作为当前索引的实体，下一个词的起始索引要大于最长实体的结束索引
    while i < (len(result5) - 1):
        # print("i=",i, result5[i])
        indexNew = result5[i][1]
        # 当前实体索引小于上一实体的结束索引，直接略过，判断下一实体
        if indexNew < maxIndexAll:
            i = i + 1
            continue
        maxIndex = i
        # 训练遍历后面的实体，找到同索引的最长实体，记录实体结束索引和下一个实体的序号
        for j in range(i + 1, len(result5)):
            # print("j=",j,result5[j])
            if result5[j][1] == indexNew:
                maxIndex = j
                i = maxIndex + 1
            else:
                maxIndexAll = result5[maxIndex][2]
                i = maxIndex + 1
                # print("up i=", i, maxIndex,maxIndexNum)
                break
        # print("maxindex=",maxIndex, result5[maxIndex])
        result6[maxIndexNum] = result5[maxIndex]
        maxIndexNum += 1
    result6 = result6[0:maxIndexNum]
    # print("===============6:===========",result6)

    ner2lable(bio, des, result6, 'B', 'I')

    for nerIndex in range(len(bio)):
        f.write(des[nerIndex] + " " + bio[nerIndex] + "\n")
    f.write("\n")
f.flush()
f.close()
print('done')
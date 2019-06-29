import pandas as pd

data_path='dbo_说明书库_研究.xlsx' #打开原始文档的路径

data_exl=pd.read_excel(data_path,names=None)  #根据路径打开excel文件

data=data_exl[['适应症']]   #提取出适应症那列数据

data.to_csv('./prepareTrainSets/Indication.csv',index=None,encoding='utf-8') #保存为新的文档


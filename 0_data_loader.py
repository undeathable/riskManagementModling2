# https://github.com/taenggu0309/PPD-Modeling-Competition/blob/master/data_input.ipynb

import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import os 
os.chdir('C:/Users/Administrator/Desktop/魔镜杯数据')

f_train1 = pd.read_csv('first_train1.csv',encoding='gbk')
f_train2 = pd.read_csv('first_train2.csv',encoding='gbk')
f_train3 = pd.read_csv('first_train3.csv',encoding='gbk')
f_test1 = pd.read_csv('first_test1.csv',encoding='gb18030')
f_test2 = pd.read_csv('first_test2.csv',encoding='gbk')
f_test3 = pd.read_csv('first_test3.csv',encoding='gbk')

# 训练集和测试集合并
f_train1['sample_status'] = 'train'
f_test1['sample_status'] = 'test'
df1 = pd.concat([f_train1,f_test1],axis=0).reset_index(drop=True)
df2 = pd.concat([f_train2,f_test2],axis=0).reset_index(drop=True)
df3 = pd.concat([f_train3,f_test3],axis=0).reset_index(drop=True)

df1.head()

# 保存数据至本地
df1.to_csv('C:/Users/Administrator/Desktop/魔镜杯数据/data_input1.csv',encoding='gb18030',index=False)
df2.to_csv('C:/Users/Administrator/Desktop/魔镜杯数据/data_input2.csv',encoding='gb18030',index=False)
df3.to_csv('C:/Users/Administrator/Desktop/魔镜杯数据/data_input3.csv',encoding='gb18030',index=False)

import numpy as np
import pandas as pd 
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os 
os.chdir('C:/Users/Administrator/Desktop/魔镜杯数据')

'''
用户数据表
'''
# 导入data_EDA_clean处理过的数据
data1 = pd.read_csv('data1_clean.csv',encoding='gbk')
# 导入变量明细表
var_info = pd.read_csv('var_info.csv',encoding='gbk')
base_col = list(data1.columns)
var_info2 = var_info[var_info.变量名称.isin(base_col)].reset_index(drop=True)
var_info2.变量类型.value_counts() 

'''
类别型变量
'''
# 类别型变量的描述性分析
cate_col = list(var_info2[var_info2.变量类型=='Categorical'].变量名称)
# 数值型类别变量的desc
data1.loc[:,cate_col].describe().T.assign(nuniq = data1.loc[:,cate_col].apply(lambda x:x.nunique()),
                                          missing_pct = data1.loc[:,cate_col].apply(lambda x:(len(x)-x.count())/len(x)))
# 先对所有字符型变量作去空格处理
for col in data1.select_dtypes(include='O').columns:
    data1[col] = data1[col].map(lambda x:str(x).strip())
## 省份
# 计算各省份违约率
def plot_pro_badrate(df,col):
    group = df.groupby(col)
    df = pd.DataFrame()
    df['total'] = group.target.count()
    df['bad'] = group.target.sum()
    df['badrate'] = df['bad']/df['total']
    # 筛选出违约率排名前5的省份
    print(df.sort_values('badrate',ascending=False).iloc[:5,:])
# 户籍地址
plot_pro_badrate(data1,'UserInfo_19')
# 西藏自治区的人数太少，不具有参考价值，剔除后再计算
plot_pro_badrate(data1[~(data1.UserInfo_19=='西藏自治区')],'UserInfo_19')
# 居住地址
plot_pro_badrate(data1,'UserInfo_7')
# 户籍省份的二值化衍生
data1['is_tianjin_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='天津市' else 0,axis=1)
data1['is_shandong_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='山东省' else 0,axis=1)
data1['is_jilin_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='吉林省' else 0,axis=1)
data1['is_sichuan_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='四川省' else 0,axis=1)
data1['is_heilongj_userinfo19'] = data1.apply(lambda x:1 if x.UserInfo_19=='黑龙江省' else 0,axis=1)
# 居住地址省份的二值化衍生
data1['is_tianjin_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='天津' else 0,axis=1)
data1['is_shandong_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='山东' else 0,axis=1)
data1['is_sichuan_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='四川' else 0,axis=1)
data1['is_hunan_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='湖南' else 0,axis=1)
data1['is_jilin_userinfo7'] = data1.apply(lambda x:1 if x.UserInfo_7=='吉林' else 0,axis=1)
# 户籍省份和居住地省份不一致衍生
data1.UserInfo_19.unique()
data1.UserInfo_7.unique()
# 将UserInfo_19改成和居住地址省份相同的格式
UserInfo_19_change = []
for i in data1.UserInfo_19:
    if i=='内蒙古自治区' or i=='黑龙江省':
        j = i[:3]
    else:
        j=i[:2]
    UserInfo_19_change.append(j)
is_same_province=[]
# 判断UserInfo_7和UserInfo_19是否一致
for i,j in zip(data1.UserInfo_7,UserInfo_19_change):
    if i==j:
        a = 1
    else:
        a = 0
    is_same_province.append(a)
    
data1['is_same_province'] = is_same_province
# 删除原有的变量
data1 = data1.drop(['UserInfo_19','UserInfo_7'],axis=1)
data1.shape
## 运营商
# 将运营商信息转换为哑变量
data1 = data1.replace({'UserInfo_9':{'中国移动':'china_mobile',
                                     '中国电信':'china_telecom',
                                     '中国联通':'china_unicom',
                                     '不详':'operator_unknown'}})
oper_dummy = pd.get_dummies(data1.UserInfo_9)
data1 = pd.concat([data1,oper_dummy],axis=1)
# 删除原变量
data1 = data1.drop(['UserInfo_9'],axis=1)
data1.shape
## 城市
# 计算4个城市特征的非重复项计数，观察是否有数据异常
for col in ['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20']:
    print('{}:{}'.format(col,data1[col].nunique()))
    print('\t')
# UserInfo_8相对其他特征nunique较大，发现有些城市有"市"，有些没有，需要做一下清洗
print(data1.UserInfo_8.unique()[:50])
# UserInfo_8清洗处理，处理后非重复项计数减小到400
data1['UserInfo_8']=[s[:-1] if s.find('市')>0 else s[:] for s in data1.UserInfo_8] 
data1.UserInfo_8.nunique()
# 根据xgboost变量重要性的输出吧对城市作二值化衍生
data1_temp1 = data1[['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20','target']]
area_list=[]
# 将四个城市变量都做亚编码处理
for col in data1_temp1:
    dummy_df = pd.get_dummies(data1_temp1[col])
    dummy_df = pd.concat([dummy_df,data1_temp1['target']],axis=1)
    area_list.append(dummy_df)

df_area1 = area_list[0]
df_area2 = area_list[1]
df_area3 = area_list[2]
df_area4 = area_list[3]
# 用xgboost建模
from xgboost.sklearn import XGBClassifier
x_area1 = df_area1.drop(['target'],axis=1)
y_area1 = df_area1['target']
x_area2 = df_area2.drop(['target'],axis=1)
y_area2 = df_area2['target']
x_area3 = df_area3.drop(['target'],axis=1)
y_area3 = df_area3['target']
x_area4 = df_area4.drop(['target'],axis=1)
y_area4 = df_area4['target']
xg_area1 = XGBClassifier(random_state=0).fit(x_area1,y_area1)
xg_area2 = XGBClassifier(random_state=0).fit(x_area2,y_area2)
xg_area3 = XGBClassifier(random_state=0).fit(x_area3,y_area3)
xg_area4 = XGBClassifier(random_state=0).fit(x_area4,y_area4)
# 输出变量的重要性
from xgboost import plot_importance
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
plot_importance(xg_area1,ax=ax1,max_num_features=10,height=0.4)
plot_importance(xg_area2,ax=ax2,max_num_features=10,height=0.4)
plot_importance(xg_area3,ax=ax3,max_num_features=10,height=0.4)
plot_importance(xg_area4,ax=ax4,max_num_features=10,height=0.4)
# 城市变量的二值化
data1['is_zibo_UserInfo2'] = data1.apply(lambda x:1 if x.UserInfo_2=='淄博' else 0,axis=1)
data1['is_chengdu_UserInfo2'] = data1.apply(lambda x:1 if x.UserInfo_2=='成都' else 0,axis=1)
data1['is_yantai_UserInfo2'] = data1.apply(lambda x:1 if x.UserInfo_2=='烟台' else 0,axis=1)

data1['is_zibo_UserInfo4'] = data1.apply(lambda x:1 if x.UserInfo_4=='淄博' else 0,axis=1)
data1['is_chengdu_UserInfo4'] = data1.apply(lambda x:1 if x.UserInfo_4=='成都' else 0,axis=1)
data1['is_weifang_UserInfo4'] = data1.apply(lambda x:1 if x.UserInfo_4=='潍坊' else 0,axis=1)

data1['is_zibo_UserInfo8'] = data1.apply(lambda x:1 if x.UserInfo_8=='淄博' else 0,axis=1)
data1['is_chengdu_UserInfo8'] = data1.apply(lambda x:1 if x.UserInfo_8=='成都' else 0,axis=1)
data1['is_shantou_UserInfo8'] = data1.apply(lambda x:1 if x.UserInfo_8=='汕头' else 0,axis=1)

data1['is_zibo_UserInfo20'] = data1.apply(lambda x:1 if x.UserInfo_20=='淄博市' else 0,axis=1)
data1['is_chengdu_UserInfo20'] = data1.apply(lambda x:1 if x.UserInfo_20=='成都市' else 0,axis=1)
data1['is_weifang_UserInfo20'] = data1.apply(lambda x:1 if x.UserInfo_20=='潍坊市' else 0,axis=1)


# 将四个城市变量改成同一的格式
data1['UserInfo_20'] = [i[:-1] if i.find('市')>0 else i[:] for i in data1.UserInfo_20]
# 城市变更次数变量衍生
city_df = data1[['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20']]
city_change_cnt =[]
for i in range(city_df.shape[0]):
    a = list(city_df.iloc[i])
    city_count = len(set(a))
    city_change_cnt.append(city_count)
data1['city_change_cnt'] = city_change_cnt
# 删除原变量
data1 = data1.drop(['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20'],axis=1)
data1.shape

## 微博
# 将字符型的nan转为众数
for col in ['WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']:
    data1 = data1.replace({col:{'nan':np.nan}})
# 将缺失填充为众数
for col in ['WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']:
    data1[col] = data1[col].fillna(data1[col].mode()[0])
# 微博变量的哑变量处理
data1['WeblogInfo_19'] = ['WeblogInfo_19_'+s for s in data1.WeblogInfo_19]
data1['WeblogInfo_21'] = ['WeblogInfo_21_'+s for s in data1.WeblogInfo_21]
for col in ['WeblogInfo_19','WeblogInfo_21']:
    dummy_df = pd.get_dummies(data1[col])
    data1 = pd.concat([data1,dummy_df],axis=1)
# 删除原变量
data1 = data1.drop(['WeblogInfo_19','WeblogInfo_21','WeblogInfo_20'],axis=1)
data1.shape

'''
数值型变量
'''
# 数值型变量的缺失率分布
import missingno
num_col = list(var_info2[var_info2.变量类型=='Numerical'].变量名称)
missingno.bar(data1.loc[:,num_col])
# 数值型变量的描述性分析
num_desc = data1.loc[:,num_col].describe().T.assign(nuniq = data1.loc[:,num_col].apply(lambda x:x.nunique()),\
                                         misssing_pct  =data1.loc[:,num_col].apply(lambda x:(len(x)-x.count())/len(x)))\
                              .sort_values('nuniq')
num_desc.head(10)
## 排序特征衍生
num_col2 = [col for col in num_col if col!='target']
# 筛选出只有数值型变量的数据集
num_data = data1.loc[:,num_col2]
# 排序特征衍生
for col in num_col2:
    num_data['rank'+col] = num_data[col].rank(method='max')/num_data.shape[0]
# 将排序特征转为单独的数据集
rank_col = [col for col in num_data.columns if col not in num_col2]
rank_df = num_data.loc[:,rank_col]
## periods变量衍生
# 生成只包含periods的临时表
periods_col = [i for i in num_col2 if i.find('Period')>0]
periods_col2 = periods_col+['target']
periods_data = data1.loc[:,periods_col2]
# 观察包含period1所有字段的数据，发现字段之间量级差异比较大，可能代表不同的含义，不适合做衍生
periods1_col = [col for col in periods_col if col.find('Period1')>0]
periods_data.loc[:,periods1_col].head()
# 观察后缀都为1的字段，发现字段数据的量级基本一致，可以对其做min,max,avg等统计值的衍生
period_1_col=[]
for i in range(0,102,17):
    col = periods_col[i]
    period_1_col.append(col)
periods_data.loc[:,period_1_col].head()
p_num_col=[]
# 将Period变量按照后缀数字存储成嵌套列表
for i in range(0,17,1):
    p_col=[]
    for j in range(i,102,17):
        col = periods_col[j]
        p_col.append(col)
    p_num_col.append(p_col)
# min,max,avg等统计值的衍生，并将衍生后的特征存成单独的数据集
periods_data = periods_data.fillna(0)
periods_fea_data=pd.DataFrame()
for j,p_list in zip(range(1,18,1),p_num_col):
    p_data = periods_data.loc[:,p_list]
    period_min=[]
    period_max=[]
    period_avg=[]
    for i in range(periods_data.shape[0]):
        a = p_data.iloc[i]
        period_min.append(np.min(a))
        period_max.append(np.max(a))
        period_avg.append(np.average(a))
    periods_fea_data['periods_'+str(j)+'_min'] = period_min
    periods_fea_data['periods_'+str(j)+'_max'] = period_max
    periods_fea_data['periods_'+str(j)+'_avg'] = period_avg
# 保存特征衍生后的数据集至本地
data1.to_csv('../魔镜杯数据/data1_process.csv',encoding='gb18030',index=False)
rank_df.to_csv('../魔镜杯数据/rank_feature.csv',encoding='gbk',index=False)
periods_fea_data.to_csv('../魔镜杯数据/periods_feature.csv',encoding='gbk',index=False)

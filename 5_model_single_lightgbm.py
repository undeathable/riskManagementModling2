import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
plt.style.use('ggplot')
import seaborn as sns 
import os 
os.chdir('C:/Users/Administrator/Desktop/魔镜杯数据')
import warnings 
warnings.filterwarnings('ignore')

import lightgbm as lgb 
from lightgbm import plot_importance 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import score_card as sc

df  = pd.read_csv('feature_select_data1.csv',encoding='gb18030')
df.head()

'''
sklearn版本的lightgbm
'''
# 默认参数模型
x_train = df[df.sample_status=='train'].drop(['Idx','sample_status','target'],axis=1)
x_test = df[df.sample_status=='test'].drop(['Idx','sample_status','target'],axis=1)
y_train = df[df.sample_status=='train']['target']
y_test = df[df.sample_status=='test']['target']

import time
start = time.time()
lgb_sklearn = lgb.LGBMClassifier(random_state=0).fit(x_train,y_train)
end = time.time()
print('运行时间为{}秒'.format(round(end-start,0)))

# 默认参数模型的AUC
lgb_sklearn_pre = lgb_sklearn.predict_proba(x_test)[:,1]
sc.plot_roc(y_test,lgb_sklearn_pre)

lgb_sklearn.get_params()

'''
原生版本
'''
# 原生的lightgbm
lgb_train = lgb.Dataset(x_train,y_train)
lgb_test = lgb.Dataset(x_test,y_test,reference=lgb_train)
lgb_origi_params = {'boosting_type':'gbdt',
              'max_depth':-1,
              'num_leaves':31,
              'bagging_fraction':1.0,
              'feature_fraction':1.0,
              'learning_rate':0.1,
              'metric': 'auc'}
start = time.time()
lgb_origi = lgb.train(train_set=lgb_train,
                      early_stopping_rounds=10,
                      num_boost_round=400,
                      params=lgb_origi_params,
                      valid_sets=lgb_test)
end = time.time()
print('运行时间为{}秒'.format(round(end-start,0)))

# 原生的lightgbm的AUC
lgb_origi_pre = lgb_origi.predict(x_test)
sc.plot_roc(y_test,lgb_origi_pre)

'''
调参
'''
# 确定最大迭代次数，学习率设为0.1 
base_parmas={'boosting_type':'gbdt',
             'learning_rate':0.1,
             'num_leaves':40,
             'max_depth':-1,
             'bagging_fraction':0.8,
             'feature_fraction':0.8,
             'lambda_l1':0,
             'lambda_l2':0,
             'min_data_in_leaf':20,
             'min_sum_hessian_inleaf':0.001,
             'metric':'auc'}
cv_result = lgb.cv(train_set=lgb_train,
                   num_boost_round=200,
                   early_stopping_rounds=5,
                   nfold=5,
                   stratified=True,
                   shuffle=True,
                   params=base_parmas,
                   metrics='auc',
                   seed=0)
print('最大的迭代次数: {}'.format(len(cv_result['auc-mean'])))
print('交叉验证的AUC: {}'.format(max(cv_result['auc-mean'])))

# num_leaves ，步长设为5
param_find1 = {'num_leaves':range(30,60,5)}
cv_fold = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
start = time.time()
grid_search1 = GridSearchCV(estimator=lgb.LGBMClassifier(learning_rate=0.1,
                                                         n_estimators = 51,
                                                         max_depth=-1,
                                                         min_child_weight=0.001,
                                                         min_child_samples=20,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0),
                             cv = cv_fold,
                             n_jobs=-1,
                             param_grid = param_find1,
                             scoring='roc_auc')
grid_search1.fit(x_train,y_train)
end = time.time()
print('运行时间为:{}'.format(round(end-start,0)))
print(grid_search1.grid_scores_)
print('\t')
print(grid_search1.best_params_)
print('\t')
print(grid_search1.best_score_)

# num_leaves,步长设为2 
param_find2 = {'num_leaves':range(26,34,2)}
grid_search2 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         min_child_weight=0.001,
                                                         min_child_samples=20,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0
                                                         ),
                            cv=cv_fold,
                            n_jobs=-1,
                            scoring='roc_auc',
                            param_grid = param_find2)
grid_search2.fit(x_train,y_train)
print(grid_search2.grid_scores_)
print('\t')
print(grid_search2.best_params_)
print('\t')
print(grid_search2.best_score_)

# 确定num_leaves 为30 ，下面进行min_child_samples 和 min_child_weight的调参，设定步长为5
param_find3 = {'min_child_samples':range(15,35,5),
               'min_child_weight':[x/1000 for x in range(1,4,1)]}
grid_search3 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         num_leaves=30,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0
                                                         ),
                            cv=cv_fold,
                            scoring='roc_auc',
                            param_grid = param_find3,
                            n_jobs=-1)
start = time.time()
grid_search3.fit(x_train,y_train)
end = time.time()
print('运行时间:{} 秒'.format(round(end-start,0)))
print(grid_search3.grid_scores_)
print('\t')
print(grid_search3.best_params_)
print('\t')
print(grid_search3.best_score_)

# 确定min_child_weight为0.001，min_child_samples为20,下面对subsample和colsample_bytree进行调参
param_find4 = {'subsample':[x/10 for x in range(5,11,1)],
               'colsample_bytree':[x/10 for x in range(5,11,1)]}
grid_search4 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         min_child_samples=20,
                                                         min_child_weight=0.001,
                                                         num_leaves=30,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8,
                                                         reg_lambda=0,
                                                         reg_alpha=0
                                                         ),
                            cv=cv_fold,
                            scoring='roc_auc',
                            param_grid = param_find4,
                            n_jobs=-1)
start = time.time()
grid_search4.fit(x_train,y_train)
end = time.time()
print('运行时间:{} 秒'.format(round(end-start,0)))
print(grid_search4.grid_scores_)
print('\t')
print(grid_search4.best_params_)
print('\t')
print(grid_search4.best_score_)

param_find5 = {'reg_lambda':[0.001,0.01,0.03,0.08,0.1,0.3],
               'reg_alpha':[0.001,0.01,0.03,0.08,0.1,0.3]}
grid_search5 = GridSearchCV(estimator=lgb.LGBMClassifier(estimator=51,
                                                         learning_rate=0.1,
                                                         min_child_samples=20,
                                                         min_child_weight=0.001,
                                                         num_leaves=30,
                                                         subsample=0.5,
                                                         colsample_bytree=0.6,
                                                         ),
                            cv=cv_fold,
                            scoring='roc_auc',
                            param_grid = param_find5,
                            n_jobs=-1)
start = time.time()
grid_search5.fit(x_train,y_train)
end = time.time()
print('运行时间:{} 秒'.format(round(end-start,0)))
print(grid_search5.grid_scores_)
print('\t')
print(grid_search5.best_params_)
print('\t')
print(grid_search5.best_score_)

# 将最佳参数再次带入cv函数，设定学习率为0.005
best_params = {
    'boosting_type':'gbdt',
    'learning_rate':0.005,
    'num_leaves':30,
    'max_depth':-1,
    'bagging_fraction':0.5,
    'feature_fraction':0.6,
    'min_data_in_leaf':20,
    'min_sum_hessian_in_leaf':0.001,
    'lambda_l1':0.3,
    'lambda_l2':0.03,
    'metric':'auc'
}

best_cv = lgb.cv(train_set=lgb_train,
                 early_stopping_rounds=5,
                 num_boost_round=2000,
                 nfold=5,
                 params=best_params,
                 metrics='auc',
                 stratified=True,
                 shuffle=True,
                 seed=0)
print('最佳参数的迭代次数: {}'.format(len(best_cv['auc-mean'])))
print('交叉验证的AUC: {}'.format(max(best_cv['auc-mean'])))

lgb_single_model = lgb.LGBMClassifier(n_estimators=900,
                                learning_rate=0.005,
                                min_child_weight=0.001,
                                min_child_samples = 20,
                                subsample=0.5,
                                colsample_bytree=0.6,
                                num_leaves=30,
                                max_depth=-1,
                                reg_lambda=0.03,
                                reg_alpha=0.3,
                                random_state=0)
lgb_single_model.fit(x_train,y_train)
pre = lgb_single_model.predict_proba(x_test)[:,1]
print('lightgbm单模型的AUC：{}'.format(metrics.roc_auc_score(y_test,pre)))
sc.plot_roc(y_test,pre)

# general model for risk management model - data preprocess part
'''
1. 打印样本分布 ==> 数据的平衡性
2. 打印/可视化缺失值，删除高缺失值率.08的特征变量，删除高缺失值率样本，少量的填充为众数
3. 打印并删除单值占比.09高/方差低的特征变量
4. 类别变量 - 计算各个类别的bad rate，再根据衍生为二值化特征，分母人数少的可以剔除（省份）
5. 类别变量 - 利用xgboost对dummy变量做特征重要性排序，将前N的做二值化衍生（城市）
6. 类别变量 - 种类少的特征直接做dummy变量衍生为特征（运营商种类）
7. 数值特征 - 
8. 特征筛选 - 拆分训练测试集并训练10个xgboost模型，取feature_important的平均，剔除特征重要性为0和比较弱的变量
9. 模型建立 - 
'''

## target variable

## categorial variable

## numerical variable

# This is a sample Python script.
import numpy as np
import pandas as pd
# from scipy import stats
# import datetime
import toad
# import importlib
# import matplotlib.pyplot as plt
import datetime
import networkx as nx
from sm3utils import sm3
from sklearn.impute import SimpleImputer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# from feature_engine.discretisation import DecisionTreeDiscretiser
# from feature_engine import discretisation
# import scorecardpy as sc
# from toad.metrics import KS, AUC
# import scipy.stats import t
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pypmml import Model, ModelElement, Parameter
from pypmml.evaluator import evaluate_model
import pandas as pd
import pypmml

def print_hi(name):
    df = pd.read_csv(f'C:/Users/liubi/tmp_122101.csv')
    df.fillna(in)
    pd.read_csv(use)
    df['dateback'] = df['dateback'].apply(
        lambda x: datetime.datetime.strptime(str(x), '%Y/%m/%d'))
    dt_set = list(set(df['dateback']))
    dt_min = df['dateback'].min()
    dt_max = df['dateback'].max()
    print("dt_min", dt_min)
    print("dt_max", dt_max)
    dt_current = dt_min
    for i in range(500):
        if dt_current not in dt_set:
            print(dt_current)
        dt_current = dt_current + datetime.timedelta(days=1)
        if dt_current >= dt_max:
            break
    # pd.Index
    # # Use a breakpoint in the code line below to debug your script.
    # # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # print(1)
    # print(np.array([1, 2, 3, 4]))
    # df_label_bad_res = pd.DataFrame(
    #     columns=['sample_id', 'data_flag', 'month', 'pos_count', 'neg_count', 'total_count', 'pos_rate'])
    # df_label_bad_res = df_label_bad_res.append(
    #     {'sample_id', 'data_flag', 'month', 'pos_count', 'neg_count', 'total_count', 'pos_rate'}, ignore_index=True)
    # print(df_label_bad_res.head())
    # df_label_bad_res.to_csv()
    # df_label_bad_res['sample_id']
    # importlib.import_module()
    # pd.Series().to
    # np.log
    # list = np.array().__abs__()
    # toad
    # plt.xlabel()
    # pd.DataFrame().sort_values(as)
    # toad.quality()
    # toad.stats
    # pd.read_csv()
    # pd.DataFrame().sort_values
    # plt.xticks(rotation=70)
    # stats.ttest_rel()
    # str(123).re
    # pd.merge().reset_index().rename(in)
    # prinstr('123')
    # G = nx.Graph()
    # G.add_weighted_edges_from()
    # toad.metrics.PSI
    # train_ids = np.array([1,2,2,1,4,5,10])
    # le = LabelEncoder()
    # oe = OrdinalEncoder()
    # train_ids_le = le.fit_transform(train_ids)
    # train_ids_oe = oe.fit_transform(train_ids)
    # print(train_ids_le)
    # print(train_ids_oe)
    # arr = np.array()
    # pd.DataFrame().fillna(inp)
    # sc.woebin
    # pd.DataFrame().index.to
    # toad.quality()
    # toad.stats._IV()
    # discretizer = DecisionTreeDiscretiser(cv=3, scoring='chi2', variables=['Age', 'Income'], regression=)
    # # 使用 fit_transform 对数据进行分箱
    # X_train_discretized = discretizer.fit_transform(X_train, y_train, )
    # pd.Series().sort_values(ascending=False)
    # pd.DataFrame().corrwith(method=)
    # G = nx.Graph()
    # G.add_node(1)
    # G.add_node(2)
    # G.add_node(3)
    # G.add_node(4)
    # G.add_edge(1, 2)
    # G.add_edge(2, 3)
    # G.add_edge(3, 4)
    # # 节点1的聚集系数
    # print(nx.clustering(G, 1))
    # # 节点的介数中心性
    # b = nx.betweenness_centrality(G)
    # for node, centrality in b.items():
    #     print(node, centrality)
    # # 节点的度
    # degrees = dict(G.degree())
    # for node, degree in degrees.items():
    #     print(node, degree)
    # # 节点的特征向量中心性
    # eigenvector_centralities = nx.eigenvector_centrality(G)
    # for node, centrality in eigenvector_centralities.items():
    #     print(node, centrality)
    # # 节点的紧密中心性
    # closeness_centralities = nx.closeness_centrality(G)
    # for node, centrality in closeness_centralities.items():
    #     print(node, centrality)
    # # 节点的pagerank重要性 - 可调参 damping factor、最大迭代系数
    # pagerank_values = nx.pagerank(G)
    # for node, pagerank in pagerank_values.items():
    #     print(node, pagerank)
    # from sm3utils import sm3
    # gen = sm3()
    # gen.update(b'abc')
    # result = gen.hexdigest()
    # print("result=", result)
    # SimpleImputer(fill_value=)
    # pd.read_csv()
    # model = RandomForestClassifier(n_jobs=-1, class_weight='unbalanced')
    # model_selector = BorutaPy(model,verbose=2)
    # model = LGBMClassifier()
    # model.best_score_()
    # pd.DataFrame(co)
    pd.DataFrame().

    # result= 66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0
    # pd.DataFrame().rolling()
    return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

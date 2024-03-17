from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.base import BaseEstimator, TransformerMixin


# 自定义转换器类
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 自定义逻辑
        transformed_data = X * 2
        return transformed_data


# 创建 Pipeline
pipeline = PMMLPipeline([
    ("custom", CustomTransformer()),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])
if __name__ == '__main__':
    print('PyCharm')
    # 保存为 PMML 文件
    sklearn2pmml(pipeline, "pipeline_model.pmml", with_repr=True)


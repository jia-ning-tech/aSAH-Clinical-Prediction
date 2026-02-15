from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model_pipeline(model_type='xgboost'):
    """预留文献中提到的6种模型接口"""
    models = {
        'lr': LogisticRegression(),
        'rf': RandomForestClassifier(),
        'xgboost': XGBClassifier()
    }
    return models.get(model_type)

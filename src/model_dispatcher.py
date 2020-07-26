# Models #
from sklearn import ensemble
import xgboost as xgb
from sklearn import linear_model

models = {

    "xgb" : xgb.XGBClassifier(
            colsample_bytree = 0.7,
            eta = 0.1,
            gamma = 0.7,
            max_depth = 9,
            min_child_weight = 3,
            reg_alpha = 0.1,
            reg_lambda = 0.1,
            subsample = 0.7),

    "rf" : ensemble.RandomForestClassifier()

    
}


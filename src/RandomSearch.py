import numpy as np
import pandas as pd
import config
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb



if __name__ == "__main__":

    # data
    df = pd.read_csv(config.TRAINING_FILE)

    # features
    features = ["ram", "battery_power", "px_height", "px_width", "mobile_wt", "int_memory",
                "talk_time","pc","clock_speed","fc","n_cores","size_screen",
                "px_res", "Mobile_wt_bin3", "m_dep_bin3","mobile_g"]
    # predictors
    X = df[features].values

    # target variable
    y = df.price_range.values

    # model 
    classifier = xgb.XGBClassifier()

    # parameters
    param_grid = {
        "eta" : [0.01,0.015,0.025,0.05,0.1],
        "gamma" : [0.1,0.3,0.5,0.7,0.9,1.0],
        "max_depth" : [3,5,9,12],
        "min_child_weight" : [1,3,5],
        "subsample" : [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree" : [0.6, 0.7, 0.8, 0.9],
        "reg_lambda" : [0.01,0.1,1.0],
        "reg_alpha" : [0,0.1,0.5,1.0]
    }

    # random search to find the best parameters
    model = model_selection.RandomizedSearchCV(
        estimator = classifier,
        param_distributions = param_grid,
        n_iter = 60,
        scoring = "accuracy",
        verbose = 10,
        n_jobs = 1,
        cv = 5
    )

    # fit the model
    model.fit(X,y)
    print(f"Best score : {model.best_score_}")

    print("Best parameters set :")
    # choose the best parameters
    best_parameters = model.best_estimator_.get_params()

    # display parameters
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name} : {best_parameters[param_name]}")
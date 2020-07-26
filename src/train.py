import pandas as pd
import numpy as np
import config
import model_dispatcher
import joblib
import os
import argparse
from learning_curves import plot_learning_curve

from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble
from sklearn import feature_selection
import matplotlib.pyplot as plt
from Feature_Importance_Selection import plot_feature_importance


def run(fold,model):
    # import training data with fold
    df = pd.read_csv(config.TRAINING_FILE)

    

    # Features to train our model on.
    # Those features are chosen based on the feature importance of
    # RandomForestClassifier() Cf. Feature_importance_Selection.py & Cf. model_dispatcher.py
    # In order to not overfit the data,we selected features from RandomForestClassifier()
    # and those features selected are used to train Xgboost model
    features = ["ram", "battery_power", "px_height", "px_width", "mobile_wt", "int_memory",
                "talk_time","pc","clock_speed","fc","n_cores","size_screen",
                "px_res", "Mobile_wt_bin3", "m_dep_bin3","mobile_g"]

    
    ## Split data into training/val data
    # training data
    df_train = df[df.kfold != fold].reset_index(drop = True)
    # validation data
    df_valid = df[df.kfold == fold].reset_index(drop = True)


    # drop price_range column and kfold
    # for training data
    x_train = df_train[features]
    y_train = df_train.price_range

    # for validation data
    x_valid = df_valid[features]
    y_valid = df_valid.price_range

    # Initialize our model
    clf = model_dispatcher.models[model]

    # fit our model on the training data
    clf.fit(x_train.values,y_train.values)
    
    # make predictions on the validation dataset
    valid_preds = clf.predict(x_valid.values)

    # evaluation metrics using accuracy score
    # In our case we don't have skewed class, so accuracy is a good metric
    # to evaluate our model
    accuracy = metrics.accuracy_score(y_valid.values,valid_preds)

    # Print accuracy of each fold
    print(f"Fold = {fold} , Accuracy = {accuracy}")

    # plot graph feature importance
    plot_feature_importance(df_train,model)

    # save the model
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f"{model}_{fold}.bin"))




if __name__ == "__main__":

    # initialize Argument ParserClass of argparse
    parser = argparse.ArgumentParser()

    # add argument
    parser.add_argument(
        "--fold",type = int
    )
    # add argument
    parser.add_argument(
        "--model",type = str
    )

    # read arg from command line
    args = parser.parse_args()

    # run our model 
    run(fold = args.fold, model = args.model)




    
    



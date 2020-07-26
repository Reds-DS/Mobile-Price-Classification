import pandas as pd
import numpy as np

from sklearn import model_selection


if __name__ == "__main__":

    # import training data
    df = pd.read_csv("../input/Mobile_train.csv")

    # Create new column "kfold" and assign value -1 at this stage
    df["kfold"] = -1

    # Randomize data
    df = df.sample(frac = 1).reset_index(drop = True)

    # Initialize StratifiedKFold Object
    Stratif_KFold = model_selection.StratifiedKFold(n_splits = 5)

    # extract target from df
    y = df.price_range.values

    # assign each validation set to his fold number
    for fold,(trn_,val_) in enumerate(Stratif_KFold.split(X = df,y = y)):
        df.loc[val_,"kfold"] = fold
    

    df.to_csv("../input/Mobile_train_fold.csv",index = False)

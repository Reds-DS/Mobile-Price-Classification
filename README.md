Mobile Price Classification - Context
---------------------------------------------------
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
The objectif of this project is to predict the price range of the mobile indicating how high the price is.
For more details [[here]](https://www.kaggle.com/iabhishekofficial/mobile-price-classification)


Frameworks / Libraries
---------------------------------------------------
* `pandas` : for data processing
* `seaborn & matplotlib` : for data visualization
* `scikit-learn` : for model selection, metrics, ml algorithm
* `xgboost` : for using xgboost classifier
* `argparse` : for parsing command Cf. `run.sh`


Project Structure - Folders
----------------------------------------------------
* input i.e datasets :
       * `Mobile_train.csv` : Training data
       * `Mobile_test.csv` : Test data
       * `Mobile_train_fold.csv` : Training data with new column "kfold" (Cross-validation) `Cf.Create_folds.py`
       * `mobile_train_modif.csv` : Training data after data wrangling & data engineering `Cf. EDA.ipynb` in notebook folder
       * `mobile_test_modif.csv` :  Test data after data wrangling & data engineering `Cf. EDA.ipynb` in notebook folder
       * `mobile_test_pred.csv` : Test data with predictions using model `xgb_2.bin` in models folder

* models : 
        * `rf_{i}.bin` : RandomForestClassifier model with `i fold number`
            * This model is used to select "important features"
        * `xgb_{i}.bin` : XgboostClassifier model with `i fold number`

* notebooks : 
        * `Check_data.ipynb` : verify data type, missing values, labels, categorical data etc .. 
        * `EDA.ipynb` : exploratory data analysis

* src :
        * `config.py` : Contain global variables, to avoid hardcoding in other files.
        * `Create_folds.py` : Script for Cross-Validation using StratifiedKFold
        * `Feature_Importance_Selection.py` : Script to choose "important" features based on random_forest algorithm
        * `RandomSearch.py` : Script for parameter tuning using Random Search 
        * `model_dispatcher.py` : models used in this project
        * `train.py` : Script for training our model
        * `test.py` : Script used to predict `price range` in the new dataset
        * `run.sh` : Command to run Script with the right fold


Metrics - Score
--------------------------------------
* Metric used is `Accuracy` from sklearn.metrics 
* Score achieved in the validation dataset : `94.5%`


Run Project
--------------------------------------
* Use bash script `run.sh` by typing `sh run.sh` command in the terminal

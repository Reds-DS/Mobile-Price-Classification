import pandas as pd
import joblib
import config
import os





if __name__ == "__main__":
    # import test set
    df_test = pd.read_csv(config.TEST_FILE)

    # feature chosen for training our model
    features = ["ram", "battery_power", "px_height", "px_width", "mobile_wt", "int_memory",
                "talk_time","pc","clock_speed","fc","n_cores","size_screen",
                "px_res", "Mobile_wt_bin3", "m_dep_bin3","mobile_g"]

    # Load our classifier model
    clf = joblib.load(os.path.join(config.MODEL_OUTPUT,"xgb_2.bin"))

    # test data
    X_test = df_test[features].values

    # test prediction
    test_preds = clf.predict(X_test)

    # add prediction to df_test
    df_test["price_range"] = list(test_preds)

    # save in csv file
    df_test.to_csv(config.TEST_OUTPUT,index = False)

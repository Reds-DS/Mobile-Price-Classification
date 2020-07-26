import model_dispatcher
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(df,model):
    # get feature importances
    importances = model_dispatcher.models[model].feature_importances_

    # get the index sorted features
    idxs = np.argsort(importances)

    # col names
    col_names =  ["ram", "battery_power", "px_height", "px_width", "mobile_wt", "int_memory",
                "talk_time","pc","sc_h","sc_w","clock_speed","fc","n_cores","size_screen",
                "px_res", "Mobile_wt_bin3", "m_dep_bin3","px_height_bin3", "px_width_bin3","mobile_g"]

    plt.title("Feature importance")
    plt.figure(figsize = (40,40))
    plt.barh(range(len(idxs)), importances[idxs], align = "center")
    plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
    plt.xlabel("RF feature importances")
    plt.show()
model_reg_path = r"C:\Users\jimja\Desktop\thesis\results_peiramatwn\1506_final_results\less\results_regression.csv"
lstm_cnn_reg_path = r"C:\Users\jimja\Desktop\thesis\results_peiramatwn\1506_final_results\less\lstm_cnn_results_regression.csv"
model_class_path = r"C:\Users\jimja\Desktop\thesis\results_peiramatwn\1506_final_results\less\results_classification.csv"
lstm_cnn_class_path = r"C:\Users\jimja\Desktop\thesis\results_peiramatwn\1506_final_results\less\lstm_cnn_results_classification.csv"

import pandas as pd




res = pd.read_csv(model_class_path)
dl = pd.read_csv(lstm_cnn_class_path)
metric = 'f1_macro' # mean_acc   std_acc  f1_macro
list = [[res[metric][1],res[metric][5]],
        [res[metric][3],res[metric][7]],
        [res[metric][2],res[metric][6]],
        [dl[metric][0],dl[metric][2]],
        [dl[metric][1],dl[metric][3]]]


print(list)

'''res = pd.read_csv(model_reg_path)
dl = pd.read_csv(lstm_cnn_reg_path)

metric = 'pval' #mean_mape std_mape   pval
list = [[res[metric][1],res[metric][5]],
        [res[metric][2],res[metric][6]],
        [res[metric][3],res[metric][7]],
        [dl[metric][0],dl[metric][2]],
        [dl[metric][1],dl[metric][3]]]


print(list)'''

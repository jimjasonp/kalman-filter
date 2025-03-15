import pandas as pd
import numpy as np
import statistics

def results_open(path,model,quantity):
    ### model = mlp,lr,dt
    ## quantity = min,mid,max
    res = pd.read_csv(path)
    result_list =[]
    if model == 'mlp': number =0
    if model == 'lr': number =1
    if model == 'dt': number =2
    
    for i in range(0,len(res)):
        if(res['Unnamed: 0'][i]) ==number:
            result_list.append(res[quantity][i])
    return result_list

#from scipy.stats import kstest

#model_results = results_open('dt','mid')

#res = kstest(model_results,'norm')



#path = r'C:\Users\jimja\Desktop\thesis\diploma_code\dokimes_unseend_data\scaled_results\none_total_mape_results'

path = r'C:\Users\jimja\Desktop\thesis\diploma_code\dokimes_unseend_data\std_results\std_total_mape_results'





'''
models = lr,mlp,dt

quantities = min,mid,max

'''

quantities_list = ['min','mid','max']
model_list  = ['lr','mlp','dt']

for model in model_list:
    for quantities in quantities_list:


        res = results_open(path,model,quantities)
        mean = statistics.mean(res)
        median = statistics.median(res)
        stdev = statistics.stdev(res)
        print('====================')
        print(model)
        print(quantities)
        print(mean,median,stdev)
        print('====================')




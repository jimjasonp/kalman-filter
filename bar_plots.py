

def regression_results_bar_charts(model_names, mape, std_devs, pvals, ylabel):
    '''
    Plots grouped bar charts comparing base and FFT models with:
    - Logarithmic y-axis
    - Proper y-axis ticks in % format
    - Enlarged figure size
    - Annotated bar labels with std and p-values
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    # Convert to numpy arrays
    mape = np.array(mape)
    std_devs = np.array(std_devs)
    pvals = np.array(pvals)

    # Avoid zero or negative MAPE (log scale can't handle it)
    mape = np.clip(mape, 0.001, None)

    model_count = len(model_names)
    x = np.arange(model_count)
    bar_width = 0.45

    # Enlarged figure size
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot bars
    bars_base = ax.bar(x - bar_width / 2, mape[:, 0], width=bar_width, label='Time',
                       color='skyblue', edgecolor='black')
    bars_fft = ax.bar(x + bar_width / 2, mape[:, 1], width=bar_width, label='FFT',
                      color='salmon', edgecolor='black')

    # Annotate bars
    for i in range(model_count):
        for j, (bar_set, color) in enumerate(zip([bars_base, bars_fft], ['skyblue', 'salmon'])):
            height = bar_set[i].get_height()
            std = std_devs[i][j]
            pval = pvals[i][j]
            label = f'{mape[i][j]:.4f} % ± {std:.4f} %\n(p value: {pval:.2e})'
            ax.text(bar_set[i].get_x() + bar_set[i].get_width() / 2,
                    height * 1.1,
                    label,
                    ha='center', va='bottom', fontsize=9)

    # Y-axis configuration
    ax.set_yscale('log')

    # Define specific y-tick values and show as %
    max_mape = np.max(mape)
    yticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    yticks = [y for y in yticks if y <= max_mape * 1.5]
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f} %'))

    # Other axis settings
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title('Regression Model Performance Comparison (Time Domain vs. FFT)')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.show()





def class_results_bar_charts(model_names, mape, std_devs, pvals, ylabel):
    '''
    The inputs are the model names, the mape values, the standard deviation values,
    the p-value values as lists for both the base models and their FFT counterparts,
    and the label on y axis.
    
    This function plots grouped bar charts comparing base and FFT models.
    '''

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    # Convert to numpy arrays
    mape = np.array(mape)
    std_devs = np.array(std_devs)
    pvals = np.array(pvals)

    # Ensure all arrays are 2D: shape (n_models, 2) -> [base, fft]
    # Example: mape[i] = [base_val, fft_val]
    model_count = len(model_names)
    x = np.arange(model_count)

    bar_width = 0.45

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot bars for base models
    bars_base = ax.bar(x - bar_width/2, mape[:, 0], width=bar_width, label='Time', 
                       color='skyblue', edgecolor='black', capsize=5)

    # Plot bars for FFT models
    bars_fft = ax.bar(x + bar_width/2, mape[:, 1], width=bar_width, label='FFT',
                      color='salmon', edgecolor='black', capsize=5)

    # Add text annotations
    for i in range(model_count):
        for j, (bar_set, color) in enumerate(zip([bars_base, bars_fft], ['skyblue', 'salmon'])):
            height = bar_set[i].get_height()
            std = std_devs[i][j]
            pval = pvals[i][j]
            label = f'{mape[i][j]:.4f} ± {std:.4f} \n(F1 macro: {pval:.4f})'
            ax.text(bar_set[i].get_x() + bar_set[i].get_width() / 2,
                    height + 0.01,
                    label,
                    ha='center', va='bottom', fontsize=10.5)

    # Set axis details
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title('Classification Model Performance Comparison (Time Domain vs. FFT)')
    ax.set_ylim(0, np.max(mape + std_devs) + 0.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


model_names = ['Random Forest', 'Linear Regression', 'MLP','LSTM','CNN']

'''
#FULL

# Each of the following should be lists of shape (n_models, 2): [[base, fft], [base, fft], ...]
mape = [[1.6504, 3.8201], [2.9995, 2.7336], [2.6718, 9.2587],[21.8589,68.9535],[6.6172,9.2939]]
std_devs = [[0.7517, 1.4301], [2.9320, 4.4888], [1.0823, 0.9734],[6.1880,15.8840],[1.6804,2.7432]]
pvals = [ [1.4665546916340996*10**-21, 4.4488*10**-15], [2.493560290130473*10*-24,1.2010*10**-21] ,[1.2847*10**-25, 1.3113529798097995*10**-26],[2.1330280929568304*10**-8,0.11869657664659342],[5.915828888060356*10**-21,9.94869130600564*10**-16]]
'''

#HALF

# Each of the following should be lists of shape (n_models, 2): [[base, fft], [base, fft], ...]
mape = [[6.63831936903785, 6.17229976033731], [2.28657517630171, 4.02404755037905], [12.49668817336225, 14.69867954056967], [23.87471307499729, 70.43526858769937], [11.09743675160603, 13.20309235749591]]
std_devs = [[4.604173534291, 2.33896090732031], [1.38425960808123, 2.44032106867265], [8.05072817784291,10.03557696130449], [13.44579031487791,18.54243389049792], [6.8816116992199, 5.42588668454853]]
pvals = [[6.427102572946627*10**-7, 5.487595845032093*10**-7], [1.4027342488550712*10**-7, 8.919660991051116*10**-10], [1.1301719445545365*10**-6, 1.5603918009162049*10**-7], [0.0006690640516289, 0.1870276571930956], [4.899238199927823*10**-6, 0.0001183019598501]]


'''
#LESS

# Each of the following should be lists of shape (n_models, 2): [[base, fft], [base, fft], ...]
mape = [[15.89936451639123, 16.4111089078518], [13.19998972797245, 8.59163680407622], [24.13024020293489, 20.70691717846247], [37.01044518122406, 45.59486682411656], [26.33134257471473, 23.72519397642028]]
std_devs = [[12.97662689505018, 15.74224694008424], [9.92854348852754, 5.99985919972548], [16.58369007200965, 10.82249481503623], [36.99037023151218, 20.18852423534969], [31.24839639723544, 9.15937238842476]]
pvals = [[0.0203092946148433, 0.0175382734350391], [0.0059489966954932, 0.0142221204060527], [0.0120502248657519, 0.0559162166228041], [0.1490430044144501, 0.200032235395937], [0.0648833657330935, 0.085450772154354]]
'''

regression_results_bar_charts(model_names, mape, std_devs, pvals, ylabel='MAPE')



model_names = ['Random Forest', 'MLP', 'SVM','LSTM','CNN']

'''
#FULL

mape = [[0.7810, 0.8105], [0.5823, 0.6937], [0.7192, 0.7945],[0.3884,0.3528],[0.7476,0.7628]]
std_devs = [[0.1031, 0.0839], [0.1038, 0.1140], [0.0835, 0.1114],[0.1257,0.09627],[0.1096,0.1045]]
pvals = [ [0.7552,0.7945], [0.5519,0.6579] ,[0.7158,0.7754],[0.3289,0.3071],[0.7260,0.7639]]
'''


#HALF

mape = [[0.6688888888888889, 0.6533333333333333], [0.4866666666666667, 0.5611111111111111], [0.5511111111111111, 0.6244444444444446], [0.1888888888888888, 0.3211111111111112], [0.5666666666666667, 0.6922222222222223]]
std_devs = [[0.2209295737179834, 0.1643393057287495], [0.1884832245212447, 0.1795914430582734], [0.1815706508072488, 0.1517632984796788], [0.1304313830295927, 0.0893667709671017], [0.2031237535574292, 0.2554613352880637]]
pvals = [[0.6383928571428571, 0.6051587301587301], [0.4173069985569985, 0.5117328042328041], [0.5152579365079364, 0.5961772486772486], [0.092024087024087, 0.2039484126984127], [0.5313308913308913, 0.6048899711399711]]


'''
#LESS

mape = [[0.4499999999999999, 0.5833333333333333], [0.4333333333333333, 0.5333333333333333], [0.4083333333333333, 0.5], [0.3249999999999999, 0.1333333333333333], [0.4083333333333333, 0.375]]
std_devs =[[0.1755942292142123, 0.1624465724134827], [0.2758824226207808, 0.1795054935711501], [0.1366768288904727, 0.2357022603955158], [0.3060727945367826, 0.135400640077266], [0.3544361719689456, 0.340036762718386]]
pvals =[[0.3336111111111111, 0.4305555555555555], [0.3347222222222222, 0.3766666666666666], [0.2925, 0.3522222222222221], [0.265, 0.0905555555555555], [0.3622222222222222, 0.325]]
'''

class_results_bar_charts(model_names, mape, std_devs, pvals, ylabel='Accuracy')
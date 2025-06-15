

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
            label = f'{mape[i][j]:.4f} ± {std:.4f} \n(F1 macro: {pval})'
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

# Each of the following should be lists of shape (n_models, 2): [[base, fft], [base, fft], ...]
mape = [[1.6504, 3.8201], [2.9995, 2.7336], [2.6718, 9.2587],[21.8589,68.9535],[6.6172,9.2939]]
std_devs = [[0.7517, 1.4301], [2.9320, 4.4888], [1.0823, 0.9734],[6.1880,15.8840],[1.6804,2.7432]]
pvals = [ [1.4665546916340996*10**-21, 4.4488*10**-15], [2.493560290130473*10*-24,1.2010*10**-21] ,[1.2847*10**-25, 1.3113529798097995*10**-26],[2.1330280929568304*10**-8,0.11869657664659342],[5.915828888060356*10**-21,9.94869130600564*10**-16]]

regression_results_bar_charts(model_names, mape, std_devs, pvals, ylabel='MAPE')

model_names = ['Random Forest', 'MLP', 'SVM','LSTM','CNN']

mape = [[0.7810, 0.8105], [0.5823, 0.6937], [0.7192, 0.7945],[0.3884,0.3528],[0.7476,0.7628]]
std_devs = [[0.1031, 0.0839], [0.1038, 0.1140], [0.0835, 0.1114],[0.1257,0.09627],[0.1096,0.1045]]
pvals = [ [0.7552,0.7945], [0.5519,0.6579] ,[0.7158,0.7754],[0.3289,0.3071],[0.7260,0.7639]]

class_results_bar_charts(model_names, mape, std_devs, pvals, ylabel='Accuracy')
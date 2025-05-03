import matplotlib.pyplot as plt
import numpy as np

def plot_model_results(model_names, accuracies, std_devs, f1_scores,ylabel):
    # Convert to numpy arrays for easier math
    accuracies = np.array(accuracies)
    std_devs = np.array(std_devs)
    f1_scores = np.array(f1_scores)

    x = np.arange(len(model_names))

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, accuracies, capsize=5, color='skyblue', edgecolor='black')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{accuracies[i]:.6f} ± {std_devs[i]:.6f}\n(p-value: {f1_scores[i]:.6f})' # bazw eite to F1: eite to p-value:
        plt.text(bar.get_x() + bar.get_width()/2, height + 0, label, # dipla sto height bazw +/- kapoion arithmo gia na einai pio omorfo
                 ha='center', va='bottom', fontsize=10)

    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title('Model performance comparison')
    plt.ylim(0, max(accuracies + std_devs) -0.02) # sto telo bazw +/- enan arithmo gia na einai pio omorfo to chart
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Example usage:
model_names = ['MLP', 'Random Forest', 'Linear Regression']
accuracies = [0.026718, 0.016504, 0.027336]
std_devs = [0.029320, 0.007517, 0.009734]
f1_scores = [0.000000, 0.000000, 0.000000]
ylabel = 'Mean Absolute Percentage Error'
plot_model_results(model_names, accuracies, std_devs, f1_scores,ylabel)

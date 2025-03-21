import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

# python hist-of-ratio.py --output_root /home/joy0921/Desktop/Dataset/MHD-3DIS/hist-of-temp

plot_labels = ['Discrete Bubble', 'SB230', 'Interconnected structure']
color_labels = ['#446e1a', '#fcbf20', '#e73332']

def load_data_from_json(filename):
    """
    Loads data from a JSON file. The file is expected to contain a list of dictionaries.
    
    Parameters:
    - filename (str): The name of the file to load the data from.
    
    Returns:
    - list: The list of dictionaries loaded from the file.
    """
    with open(filename, 'r') as file:
        return json.load(file)

def plot_histograms_from_list(data_list, output_path):
    """
    Plots histograms for each dictionary in the list, each in a different color.
    
    Parameters:
    - data_list (list): A list of dictionaries containing data values to plot.
    - output_path (str): Path to save the generated histogram image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define histogram parameters
    n_bins = 100
    x_min, x_max = 0, 1  # Adjust as needed based on your data range
    bins = np.linspace(x_min, x_max, n_bins + 1)
    
    # Use the default color cycle from matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Loop over each dictionary in the list and plot its histogram
    for idx, data_dict in enumerate(data_list):
        values = list(data_dict.values())
        color = color_labels[idx] #colors[idx % len(colors)]
        ax.hist(values, bins=bins, histtype='step', linestyle='solid',
                color=color, label=f"{plot_labels[idx]}")
    
    # Set axis labels, title, limits, and logarithmic scale for y-axis
    ax.set_xlabel("$\mathcal{R}$", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Histogram of Ratio", fontsize=14)
    ax.set_xlim(x_min, x_max)
    # ax.set_yscale('log')
    
    # Display grid lines and legend
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of ratio values from multiple datasets."
    )
    parser.add_argument("--input_root", default=".", type=str, help="Input directory")
    parser.add_argument("--output_root", default="./Dataset", type=str, help="Output directory")
    
    args = parser.parse_args()
    ratio_file = os.path.join(args.input_root, 'interconnectedness-ratio.json')
    
    # Load list of dictionaries from the JSON file
    data_list = load_data_from_json(ratio_file)
    
    output_file = os.path.join(args.output_root, 'ratio-histogram.png')
    plot_histograms_from_list(data_list, output_file)
    
    print(f"Done. Plot saved at: {output_file}")

if __name__ == "__main__":
    main()
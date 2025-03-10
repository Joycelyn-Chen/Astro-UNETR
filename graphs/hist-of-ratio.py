import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt



def load_dict_from_json(filename):
    """
    Loads a dictionary from a JSON file.
    
    Parameters:
    - filename (str): The name of the file to load the dictionary from.
    
    Returns:
    - dict: The dictionary loaded from the file.
    """
    with open(filename, 'r') as file:
        return json.load(file)

def plot_histogram_from_dict(data_dict, output_path):
    """
    Plots a histogram based on the values from the provided dictionary.
    
    Parameters:
    - data_dict (dict): Dictionary containing data values to plot.
    - output_path (str): Path to save the generated histogram image.
    """
    # Extract values from the dictionary
    values = list(data_dict.values())
    
    # Define histogram parameters
    n_bins = 100
    x_min, x_max = 0, 1  # Adjust as needed based on your data range
    bins = np.linspace(x_min, x_max, n_bins + 1)
    
    # Compute logarithmic values for plotting
    # log_values = np.log10(values)
    
    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram
    ax.hist(values, bins=bins, histtype='step', linestyle='solid', color='black')
    
    
    # Set axis labels and title
    ax.set_xlabel("$\mathcal{R}$", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Histogram of Ratio", fontsize=14)
    
    # Set axis limits and scaling
    ax.set_xlim(x_min, x_max)
    # ax.set_ylim(1e-5, 1e2)
    ax.set_yscale('log')
    
    # Display grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot the histogram of ratio values around the bubble"
    )
    parser.add_argument("--input_root", default=".", type=str, help="Input directory")
    parser.add_argument("--output_root", default="./Dataset", type=str, help="Output directory")

    
    
    args = parser.parse_args()
    ratio_file = os.path.join(args.input_root, 'interconnectedness-ratio.json')
    loaded_dict = load_dict_from_json(ratio_file)

    output_file = os.path.join(args.output_root, 'ratio-histogram.png')
    plot_histogram_from_dict(loaded_dict, output_file)

    print(f"Done. Plot saved at: {output_file}")


if __name__ == "__main__":
    main()


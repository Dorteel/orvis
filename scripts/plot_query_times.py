#!/home/user/pel_ws/pel_venv/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to process two timing result files and create a log-scaled box plot
def process_and_plot(file1, file2, column_names=("Function", "Execution Time")):
    # Load the CSV files into dataframes
    df1 = pd.read_csv(file1, header=None, names=column_names)
    df2 = pd.read_csv(file2, header=None, names=column_names)

    # Take only the last 1000 entries
    df1 = df1.tail(1000)
    df2 = df2.tail(1000)

    # Extract execution times
    times1 = df1[column_names[1]]
    times2 = df2[column_names[1]]

    # Print min, max, and mean for each file
    print(f"{file1} - Min: {times1.min():.4f}, Max: {times1.max():.4f}, Mean: {times1.mean():.4f}, Std: {times1.std():.4f}")
    print(f"{file2} - Min: {times2.min():.4f}, Max: {times2.max():.4f}, Mean: {times2.mean():.4f}, Std: {times2.std():.4f}")

    # Create a log-scaled box plot
    plt.figure(figsize=(6, 6))  # Adjust width to make the plot narrower
    plt.boxplot([np.log10(times1), np.log10(times2)], labels=['Location Query', 'Annotator Capability Query'], showmeans=True, widths=0.6)
    
    # Customize y-axis to show original seconds
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f"{10**y:.4f}" for y in yticks])

    # Update the title with the number of entries
    plt.title(f"Execution Time Comparison (Log Scale, Last 1000 Entries)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(axis='y')
    plt.show()

# Function to process two timing result files and create a violin plot
def process_and_plot_violin(file1, file2, column_names=("Function", "Execution Time")):
    # Load the CSV files into dataframes
    df1 = pd.read_csv(file1, header=None, names=column_names)
    df2 = pd.read_csv(file2, header=None, names=column_names)

    # Take only the last 1000 entries
    df1 = df1.tail(1000)
    df2 = df2.tail(1000)

    # Extract execution times
    times1 = df1[column_names[1]]
    times2 = df2[column_names[1]]

    # Print min, max, and mean for each file
    print(f"{file1} - Min: {times1.min():.4f}, Max: {times1.max():.4f}, Mean: {times1.mean():.4f}, Std: {times1.std():.4f}")
    print(f"{file2} - Min: {times2.min():.4f}, Max: {times2.max():.4f}, Mean: {times2.mean():.4f}, Std: {times2.std():.4f}")

    # Create a violin plot
    plt.figure(figsize=(6, 6))
    plt.violinplot([np.log10(times1), np.log10(times2)], showmeans=True)
    
    # Customize y-axis to show original seconds
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f"{10**y:.4f}" for y in yticks])

    # Update x-axis labels
    plt.xticks([1, 2], ['Location Query', 'Annotator Capability Query'])

    # Update the title with the number of entries
    plt.title(f"Execution Time Comparison (Log Scale, 1000 Queries)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(axis='y')
    plt.show()

# Example usage
file1 = "results/location_query_results.csv"
file2 = "results/annotator_query_results.csv"
process_and_plot(file1, file2)
# process_and_plot_violin(file1, file2)
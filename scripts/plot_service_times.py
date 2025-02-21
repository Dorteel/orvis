import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def summarize_service_data(file_path):
    """
    Reads a CSV file with service call times, computes summary statistics (mean, median,
    standard deviation, min, and max) for each service_name grouped by service_type,
    and saves the summary to a CSV file.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Ensure required columns exist
        if 'service_name' not in data.columns or 'service_type' not in data.columns:
            raise ValueError("The required columns 'service_name' and 'service_type' are missing.")

        # Melt the data for call times
        melted_data = pd.melt(
            data,
            id_vars=['service_name', 'service_type'],
            value_vars=[col for col in data.columns if col.startswith('call_')],
            var_name='Call',
            value_name='Call_Time'
        )

        # Ensure Call_Time is numeric
        melted_data['Call_Time'] = pd.to_numeric(melted_data['Call_Time'], errors='coerce')

        # Compute summary statistics grouped by service_type and service_name
        summary = melted_data.groupby(['service_type', 'service_name'])['Call_Time'].agg(
            mean='mean',
            median='median',
            std='std',
            min='min',
            max='max'
        ).reset_index()

        # Save the summary to a CSV file
        summary.to_csv('service_summary_statistics.csv', index=False)
        print("Summary statistics saved to 'service_summary_statistics.csv'.")

        # Print the summary in a readable format
        print("Summary Statistics:")
        print(summary)

    except Exception as e:
        print(f"An error occurred: {e}")

# Function to plot bar chart with error bars on a log scale
def plot_bar_chart_with_stats(file_path):
    """
    Reads a CSV file, computes statistics (mean, median, std, etc.), and plots a bar chart
    with service_name on the x-axis and mean call time on the y-axis, grouped by service_type.
    Displays error bars for standard deviation, with a logarithmic y-axis.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Ensure required columns exist
        if 'service_name' not in data.columns or 'service_type' not in data.columns:
            raise ValueError("The required columns 'service_name' and 'service_type' are missing.")

        # Melt the data for call times
        melted_data = pd.melt(
            data,
            id_vars=['service_name', 'service_type'],
            value_vars=[col for col in data.columns if col.startswith('call_')],
            var_name='Call',
            value_name='Call_Time'
        )

        # Ensure Call_Time is numeric
        melted_data['Call_Time'] = pd.to_numeric(melted_data['Call_Time'], errors='coerce')

        # Compute statistics grouped by service_type and service_name
        stats = melted_data.groupby(['service_type', 'service_name'])['Call_Time'].agg(
            mean='mean',
            median='median',
            std='std',
            min='min',
            max='max'
        ).reset_index()

        stats = stats.sort_values(by='mean', ascending=True)
        # Generate bar positions for each service_name
        unique_services = stats['service_name'].unique()
        unique_types = stats['service_type'].unique()
        x_positions = np.arange(len(unique_services))

        # Bar width and spacing
        bar_width = 0.35  # Set bar width
        offsets = np.linspace(-bar_width, bar_width, len(unique_types))

        # Create the plot with narrower width
        plt.figure(figsize=(12, 10))  # Set width to 12 and height to 10 for a narrower plot
        for idx, service_type in enumerate(unique_types):
            # Filter data for the current service_type
            subset = stats[stats['service_type'] == service_type]

            # Align x_positions with service_name indices in the subset
            service_indices = [np.where(unique_services == name)[0][0] for name in subset['service_name']]
            adjusted_positions = x_positions[service_indices] + offsets[idx]

            # Add bars with error bars for std deviation
            plt.bar(
                adjusted_positions,
                subset['mean'],
                yerr=subset['std'],  # Error bars for std deviation
                capsize=5,  # Add caps on error bars
                width=bar_width,
                label=f"{service_type}"
            )

        # Set x-axis labels and y-axis scale (logarithmic)
        plt.xticks(x_positions, unique_services, rotation=35, ha='right', fontsize=20)  # Larger x-axis labels, aligned properly
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.ylim(0.1, stats['max'].max() * 1.2)  # Log scale with padding above max value

        # Add custom log ticks
        y_ticks = np.logspace(-1, np.ceil(np.log10(stats['max'].max())), num=10)  # Logarithmic ticks
        plt.yticks(y_ticks, labels=[f"{t:.1f}" for t in y_ticks], fontsize=20)

        # Customize the plot
        plt.title('Service Execution Time comparison (Log Scale)', fontsize=20)
        plt.xlabel('', fontsize=20)
        plt.ylabel('Response time (s)', fontsize=20)
        plt.legend(
            title='Service Type',
            fontsize=16.5,  # Increased font size
            title_fontsize=18,
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99)
        )  # Legend inside the plot, on the left

        # Show the plot
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "results/service_evaluation_times.csv"  # Ensure this is the correct path to your file
    plot_bar_chart_with_stats(file_path)
    summarize_service_data(file_path)
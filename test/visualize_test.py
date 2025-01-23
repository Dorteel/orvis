import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "service_call_times.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Compute statistics
data['avg_time'] = data.iloc[:, 1:].mean(axis=1)
data['max_time'] = data.iloc[:, 1:].max(axis=1)
data['min_time'] = data.iloc[:, 1:].min(axis=1)

# Extract model type and model name for a cleaner service name
data['service_label'] = data['service_name'].apply(
    lambda x: f"{x.split('/')[2]}-{x.split('/')[3]}"
)

# Sort by average time for better visualization
data = data.sort_values(by='avg_time', ascending=False)

# Plot average, max, and min times for each service
plt.figure(figsize=(12, 6))
plt.barh(data['service_label'], data['avg_time'], color='skyblue', label='Average Time')
plt.barh(data['service_label'], data['max_time'], color='orange', alpha=0.5, label='Max Time')
plt.barh(data['service_label'], data['min_time'], color='green', alpha=0.5, label='Min Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Service Name')
plt.title('Service Call Times with 100 calls per service (Average, Max, Min)')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("service_call_times_visualization.png")
plt.show()

# Print statistics for each service
print(data[['service_name', 'avg_time', 'max_time', 'min_time']])

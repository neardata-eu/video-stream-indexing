import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
with open('results/inference.log', 'r') as file:
    lines = file.readlines()

# Parse the data
e2e_latency = []
model_inference = []
milvus_transfer = []

for line in lines[10:]:  # Skip the header
    values = line.strip().split(',')
    e2e_latency.append(float(values[0]))
    model_inference.append(float(values[1]))
    milvus_transfer.append(float(values[2]))

# Calculate the averages
avg_e2e_latency = np.mean(e2e_latency)
avg_model_inference = np.mean(model_inference)
avg_milvus_transfer = np.mean(milvus_transfer)

# Create a single bar plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the averages as a stacked bar
bar_width = 0.5
indices = np.arange(1)

p1 = ax.bar(indices, avg_e2e_latency, bar_width, label='e2e latency (ms)')
p2 = ax.bar(indices, avg_model_inference, bar_width, bottom=avg_e2e_latency, label='model inference (ms)')
p3 = ax.bar(indices, avg_milvus_transfer, bar_width, bottom=avg_e2e_latency + avg_model_inference, label='milvus transfer (ms)')

ax.set_xticks([])

# Adding labels and title
ax.set_ylabel('Time (ms)')
ax.set_title('Average e2e latency, model inference, and milvus transfer times')

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()

plt.savefig('results/inference_analysis.png')

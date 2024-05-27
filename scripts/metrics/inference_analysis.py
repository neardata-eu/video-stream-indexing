import json
import math
import matplotlib.pyplot as plt

def calculate_average_and_stddev(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    total_inference_time = 0
    total_insert_time = 0
    count = len(data)
    
    inference_times = []
    insert_times = []
    
    for entry in data:
        start = entry["start"]
        inference = entry["inference"]
        insert = entry["insert"]
        
        inference_time = (inference - start) * 1000
        insert_time = (insert - inference) * 1000
        
        inference_times.append(inference_time)
        insert_times.append(insert_time)
        
        total_inference_time += inference_time
        total_insert_time += insert_time
    
    average_inference_time = total_inference_time / count if count != 0 else 0
    average_insert_time = total_insert_time / count if count != 0 else 0
    
    # Calculate standard deviation
    inference_variance = sum((x - average_inference_time) ** 2 for x in inference_times) / count if count != 0 else 0
    insert_variance = sum((x - average_insert_time) ** 2 for x in insert_times) / count if count != 0 else 0
    
    stddev_inference_time = math.sqrt(inference_variance)
    stddev_insert_time = math.sqrt(insert_variance)
    
    print(f"Average Inference Time: {average_inference_time} ms")
    print(f"Standard Deviation of Inference Time: {stddev_inference_time} ms")
    print(f"Average Insert Time: {average_insert_time} ms")
    print(f"Standard Deviation of Insert Time: {stddev_insert_time} ms")

    # Generate plot
    labels = ['Inference Time', 'Insert Time']
    averages = [average_inference_time, average_insert_time]
    std_devs = [stddev_inference_time, stddev_insert_time]
    
    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, averages, yerr=std_devs, capsize=10, alpha=0.7, color=['blue', 'green'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (milliseconds)')
    ax.set_title('Frame Ingestion')

    plt.savefig('results/inference_analysis.png')

# Specify the path to your JSON file
file_path = 'results/inference_metrics.json'

calculate_average_and_stddev(file_path)

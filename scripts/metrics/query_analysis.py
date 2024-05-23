import json
import math
import matplotlib.pyplot as plt

def calculate_average_and_stddev(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    total_query_time = 0
    total_pravega_retrieve_time = 0
    count_query = 0
    count_pravega_retrieve = 0
    
    query_times = []
    pravega_retrieve_times = []
    
    for entry in data:
        start = entry.get("start")
        query = entry.get("query")
        pravega_retrieve = entry.get("pravega_retrieve")
        
        if start is not None and query is not None:
            query_time = query - start
            query_times.append(query_time)
            total_query_time += query_time
            count_query += 1
        
        if query is not None and pravega_retrieve is not None:
            pravega_retrieve_time = pravega_retrieve - query
            pravega_retrieve_times.append(pravega_retrieve_time)
            total_pravega_retrieve_time += pravega_retrieve_time
            count_pravega_retrieve += 1
    
    average_query_time = total_query_time / count_query if count_query != 0 else 0
    average_pravega_retrieve_time = total_pravega_retrieve_time / count_pravega_retrieve if count_pravega_retrieve != 0 else 0
    
    # Calculate standard deviation
    query_variance = sum((x - average_query_time) ** 2 for x in query_times) / count_query if count_query != 0 else 0
    pravega_retrieve_variance = sum((x - average_pravega_retrieve_time) ** 2 for x in pravega_retrieve_times) / count_pravega_retrieve if count_pravega_retrieve != 0 else 0
    
    stddev_query_time = math.sqrt(query_variance)
    stddev_pravega_retrieve_time = math.sqrt(pravega_retrieve_variance)
    
    print(f"Average query Time: {average_query_time}")
    print(f"Standard Deviation of Query Time: {stddev_query_time}")
    print(f"Average pravega_retrieve Time: {average_pravega_retrieve_time}")
    print(f"Standard Deviation of Pravega Segment Retrieval Time: {stddev_pravega_retrieve_time}")

    # Generate plot
    labels = ['Query Time', 'Pravega Segment Retrieval Time']
    averages = [average_query_time, average_pravega_retrieve_time]
    std_devs = [stddev_query_time, stddev_pravega_retrieve_time]
    
    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, averages, yerr=std_devs, capsize=10, alpha=0.7, color=['blue', 'green'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Query and Pravega Segment Retrieval Times')

    plt.savefig('results/query_analysis.png')

# Specify the path to your JSON file
file_path = 'results/query_metrics.json'

calculate_average_and_stddev(file_path)

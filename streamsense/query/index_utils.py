import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import (
    Collection,
)

from collections import defaultdict
import subprocess
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import generate_fragments


def search_global(collection_name, embedding, fields, k, accuracy, f):
    """Search the global collection for candidate streams"""
    start = time.time()
    collection = Collection(collection_name)
    collection_con = time.time()
    collection.load()
    col_load = time.time()
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=k*f, output_fields=fields)
    search_time = time.time()
    
    # Filter results
    videos = []
    counter = 0
    for hits in result:
        for hit in hits:
            counter = counter+1
            if hit.distance >= accuracy:
                videos.append((hit.collection, hit.distance))
    print(f"Total hits: {counter}")    
    filter1 = time.time()
            
    # Get unique streams
    highest_values = defaultdict(lambda: float('-inf')) 
    for key, value in videos:
        highest_values[key] = max(highest_values[key], value)
    sorted_data = sorted(highest_values.items(), key=lambda item: item[1], reverse=True)
    
    filter2 = time.time()
    
    # Print query times
    #print(f"Collection: {collection_con - start}")
    #print(f"Load: {col_load - collection_con}")
    #print(f"Search: {search_time - col_load}")
    #print(f"Filter1: {filter1 - search_time}")
    #print(f"Filter2: {filter2 - filter1}")
    
    search_times = {
        "collection": collection_con - start,
        "load": col_load - collection_con,
        "search": search_time - col_load,
        "filter1": filter1 - search_time,
        "filter2": filter2 - filter1
    }
    
    return [item[0] for item in sorted_data[:k]], search_times


def process_offset(idx, off_start, off_end, collection_name, result_path, env):
    """Export a video fragment from Pravega"""
    filename = f"{collection_name}_{idx}_{off_start}_{off_end}.h264"
    subprocess.run(['bash', '/project/streamsense/query/export.sh', collection_name, f"{result_path}/{filename}", off_start, off_end], 
                    env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gb_retrieved = os.path.getsize(f"{result_path}/{filename}") / (1024 ** 3)
    return filename, gb_retrieved


def search(milvus_client, collection_name, embedding, fields, local_k, fragment_offset, accuracy, result_path, parallelism):
    """Search a collection for similar segments and get those fragments from Pravega"""
    start_time = time.time()
    collection = Collection(collection_name)
    create_collection_time = time.time()
    collection.load()
    load_collection_time = time.time()
    
    # Search the Index
    print(f"Searching in {collection_name}")
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=local_k, output_fields=fields)
    query_time = time.time()
    
    # Filter results
    frames = []
    for hits in result:
        for hit in hits:
            frames.append((hit.pk, hit.distance))
    merged_intervals = generate_fragments(frames, fragment_offset, accuracy)
    generate_fragments_time = time.time()
    
    # Get the offsets from the bounds
    offsets = []
    for (start, end) in merged_intervals:
        offsets.append(max(start, 0))
        offsets.append(min(end, int(collection.num_entities)-1))
    bounds = milvus_client.get(collection_name=collection_name, ids=offsets, output_fields=["offset"])
    get_offsets_time = time.time()
    
    # Filter other fields out
    offset_dict = list(zip(bounds[::2], bounds[1::2]))
    offsets = []
    for (start, end) in offset_dict:
        offsets.append((start["offset"], end["offset"]))
    filter_fields_out_time = time.time()
    
    # Export fragments from Pravega using Threads
    print(f"Exporting fragments from {collection_name}")
    env = os.environ.copy()
    total_gb_retrieved = 0
    files = []
    files_and_gbs = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [
            executor.submit(process_offset, idx, off_start, off_end, collection_name, result_path, env)
            for idx, (off_start, off_end) in enumerate(offsets)
        ]
        for future in as_completed(futures):
            try:
                filename, gb_retrieved = future.result()
                files.append(filename)
                total_gb_retrieved += gb_retrieved
                files_and_gbs.append((filename, gb_retrieved))
            except Exception as e:
                print(f"An error occurred: {e}")
    export_time = time.time()
    
    # Log
    log = {
        "collection": collection_name,
        "create_collection_ms": (create_collection_time - start_time)*1000,
        "load_collection_ms": (load_collection_time - create_collection_time)*1000,
        "query_ms": (query_time - load_collection_time)*1000,
        "generate_fragments_ms": (generate_fragments_time - query_time)*1000,
        "get_offsets_ms": (get_offsets_time - generate_fragments_time)*1000,
        "filter_fields_out_ms": (filter_fields_out_time - get_offsets_time)*1000,
        "export_ms": (export_time - filter_fields_out_time)*1000,
        "total_ms": (export_time - start_time)*1000,
        "fragments_and_gbs": files_and_gbs
    }
    
    return files, total_gb_retrieved, log
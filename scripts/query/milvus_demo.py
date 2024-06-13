import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)

from PIL import Image

import numpy as np

import subprocess
import time
from collections import defaultdict
import json
from datetime import datetime
import argparse

from policies.constants import (MILVUS_HOST, MILVUS_PORT, MILVUS_NAMESPACE,
                                LOG_PATH, RESULT_PATH)
from policies.components import get_model, inference


latency_dict = {}


def count_frames(filepath):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-print_format', 'json',
        filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    frame_count = int(info['streams'][0]['nb_read_frames'])
    return frame_count


def process_directory(directory_path):
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.h264'):
            file_path = os.path.join(directory_path, filename)
            frame_count = count_frames(file_path)
            if frame_count is not None:
                results.append({"filename": filename, "frame_count": frame_count})
            else:
                raise Exception(f'Error counting frames in file {filename}')
    return results


def search_global(collection_name, embedding, fields, k):
    collection = Collection(collection_name)
    collection.load()
    
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=k*100, output_fields=fields)
    
    videos = []
    
    for hits in result:
        for hit in hits:
            videos.append((hit.collection, hit.distance))
    
    highest_values = defaultdict(lambda: float('-inf'))  # Initialize with negative infinity

    for key, value in videos:
        # Update the highest value if the current value is greater.
        highest_values[key] = max(highest_values[key], value)

    sorted_data = sorted(highest_values.items(), key=lambda item: item[1], reverse=True)
    return [item[0] for item in sorted_data[:k]]


def search(milvus_client, collection_name, embedding, fields, k, accuracy):
    collection = Collection(collection_name)
    collection.load()
    
    start = time.time()
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=k, output_fields=fields)
    search_time = time.time()
    
    hit_num = 0
    fail_num = 0
    gb_retrieved = 0
    for hits in result:
        for hit in hits:
            if (hit.distance < accuracy):
                fail_num += 1
                latency_dict["frame_search_retrieve"].append({
                    "index_search_ms": (search_time - start)*1000,
                    "pravega_retrieve_ms": 0
                })
            else:
                print(f"Processing hit {hit.pk} with distance {hit.distance}")
                bounds = milvus_client.get(collection_name=collection_name, ids=[int(hit.pk)-20, int(hit.pk)+20], output_fields=["offset"])
                get_bounds = time.time()
                
                env = os.environ.copy()
                subprocess.run(['bash', '/project/scripts/query/export.sh', collection_name, f"{collection_name}_{hit.pk}.h264", bounds[0]["offset"], bounds[1]["offset"]], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                hit_num += 1
                
                pravega_retrieve = time.time()
                latency_dict["frame_search_retrieve"].append({
                    "index_search_ms": (get_bounds - start)*1000,
                    "pravega_retrieve_ms": (pravega_retrieve-get_bounds)*1000
                })
                
                gb_retrieved += os.path.getsize(f"{RESULT_PATH}/{collection_name}_{hit.pk}.h264") / (1024 ** 3)
    return hit_num, fail_num, gb_retrieved


def main():
    parser = argparse.ArgumentParser(description='Milvus Query Demo')
    parser.add_argument('--image_path', default='/project/benchmarks/experiment3/cat_frame_ref.png')
    parser.add_argument('--global_k', default=5)
    parser.add_argument('--accuracy', default=0.9)
    args = parser.parse_args()
    
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    ## Read our query image
    latency_dict["image_path"] = args.image_path
    img = Image.open(args.image_path)
    
    ## Initialize embedding model
    print("Initializing model")
    model, device = get_model()
    start = time.time()
    embeds = inference(model, np.array(img), device)  # Get embeddings
    inference_time = time.time()
    latency_dict["inference_ms"] = (inference_time - start)*1000

    ## Search global index
    candidates = search_global("global", embeds.detach().numpy(), ["collection"], int(args.global_k))
    global_search = time.time()
    latency_dict["search_global_ms"] = (global_search - inference_time)*1000
    
    print("Candidates found:")
    print(candidates)
    latency_dict["candidates"] = candidates

    ## Perform queries
    print("Performing search")
    hit_num = 0
    fail_num = 0
    gb_retrieved = 0
    
    #collection_list = utility.list_collections()
    latency_dict["frame_search_retrieve"] = []
    for collection_name in candidates: # Search in the candidate collections
        output_fields=["offset", "pk"]
        hit, fail, gb = search(client, collection_name, embeds.detach().numpy(), output_fields, 1, float(args.accuracy))
        hit_num += hit
        fail_num += fail
        gb_retrieved += gb

    print(f"Number of coincidences found in the database: {hit_num}/{hit_num+fail_num}")
    
    latency_dict["frame_count"] = process_directory(RESULT_PATH)
    latency_dict["total_gb_retrieved"] = gb_retrieved
    
    config = {
        "image_path": args.image_path,
        "global_k": args.global_k,
        "accuracy": args.accuracy,
        "log_path": LOG_PATH,
        "result_path": RESULT_PATH,
    }
    latency_dict["config"] = config
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{LOG_PATH}/query_logs_{timestamp}.json", "w") as f:
        json.dump(latency_dict, f)

    
if __name__ == "__main__":
    main()
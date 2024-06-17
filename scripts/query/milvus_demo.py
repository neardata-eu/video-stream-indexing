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


def process_files(filenames, result_path):
    results = []
    for filename in filenames:
        if filename is not None:
            frame_count = count_frames(f"{result_path}/{filename}")
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


def generate_fragments(frames, fragment_offset=5, similarity=0.9):
    """
    Generate fragments from a list of frames
    :param frames: a list of frames with the format (frame_number, similarity)
    :param fragment_offset: a number of frames to consider before and after the frame
    :param similarity: a threshold to consider a frame as a key frame
    :return: a list of tuples with the start and end of the fragments
    """
    
    # Remove frames with similarity below the threshold
    frames = [frame for frame in frames if frame[1] >= similarity]
    
    # Generate intervals around the frames
    intervals = [(frame[0] - fragment_offset, frame[0] + fragment_offset) for frame in frames]

    # Check if the intervals overlap and merge them
    merged_intervals = []
    for start, end in sorted(intervals):
        if merged_intervals and start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
        else:
            merged_intervals.append((start, end))

    return merged_intervals


def search(milvus_client, collection_name, embedding, fields, local_k, fragment_offset, accuracy, result_path):
    collection = Collection(collection_name)
    collection.load()
    
    start = time.time()
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=local_k, output_fields=fields)
    
    frames = []
    for hits in result:
        for hit in hits:
            frames.append((hit.pk, hit.distance))
    merged_intervals = generate_fragments(frames, fragment_offset, accuracy)
    
    offsets = []
    for interval in merged_intervals:
        bounds = milvus_client.get(collection_name=collection_name, ids=[interval[0], interval[1]], output_fields=["offset"])
        offsets.append((bounds[0]["offset"], bounds[1]["offset"]))
    search_time = time.time()
    
    env = os.environ.copy()
    gb_retrieved = 0
    files = []
    for idx, (off_start, off_end) in enumerate(offsets):
        filename = f"{collection_name}_{idx}_{off_start}_{off_end}.h264"
        subprocess.run(['bash', '/project/scripts/query/export.sh', collection_name, f"{result_path}/{filename}", off_start, off_end], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        files.append(filename)
        gb_retrieved += os.path.getsize(f"{result_path}/{filename}") / (1024 ** 3)
    export_time = time.time()
    
    latency_dict["frame_search_retrieve"].append({
        "collection": collection_name,
        "search_ms": (search_time - start)*1000,
        "export_ms": (export_time - search_time)*1000,
        "fragments": files
    })
    
    return files, gb_retrieved


def main():
    parser = argparse.ArgumentParser(description='Milvus Query Demo')
    parser.add_argument('--image_path', default='/project/benchmarks/experiment3/cholec_frame_ref.png')
    parser.add_argument('--global_k', default=5)
    parser.add_argument('--local_k', default=100)
    parser.add_argument('--fragment_offset', default=10)
    parser.add_argument('--accuracy', default=0.9)
    parser.add_argument('--log_path', default=LOG_PATH)
    parser.add_argument('--result_path', default=RESULT_PATH)
    args = parser.parse_args()
    
    log_path = args.log_path
    result_path = args.result_path
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    ## Read our query image
    latency_dict["image_path"] = args.image_path
    img = Image.open(args.image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
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
    fragments = []
    
    #collection_list = utility.list_collections()
    latency_dict["frame_search_retrieve"] = []
    gb_retrieved = 0
    for collection_name in candidates: # Search in the candidate collections
        output_fields=["offset", "pk"]
        fragment, gb_retrieved = search(client, collection_name, embeds.detach().numpy(), output_fields, int(args.local_k), int(args.fragment_offset), float(args.accuracy), result_path)
        fragments.extend(fragment)
    
    latency_dict["frame_count"] = process_files(fragments, result_path)
    latency_dict["total_gb_retrieved"] = gb_retrieved
    
    config = {
        "image_path": args.image_path,
        "global_k": args.global_k,
        "local_k": args.local_k,
        "fragment_offset": args.fragment_offset,
        "accuracy": args.accuracy,
        "log_path": log_path,
        "result_path": result_path,
    }
    latency_dict["config"] = config
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{log_path}/query_logs_{timestamp}.json", "w") as f:
        json.dump(latency_dict, f)

    
if __name__ == "__main__":
    main()
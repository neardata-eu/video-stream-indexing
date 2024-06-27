import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pymilvus import (
    connections,
    MilvusClient
)

from PIL import Image

import numpy as np

import time
import json
from datetime import datetime
import argparse

from policies.constants import (MILVUS_HOST, MILVUS_PORT, MILVUS_NAMESPACE,
                                LOG_PATH, RESULT_PATH)
from policies.components import get_model, inference

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from utils import process_files
from index_utils import search, search_global


def main():
    parser = argparse.ArgumentParser(description='Milvus Query Demo')
    parser.add_argument('--image_path', default='/project/benchmarks/experiment3/cholec_frame_ref.png')
    parser.add_argument('--global_k', default=5)
    parser.add_argument('--local_k', default=100)
    parser.add_argument('--fragment_offset', default=10)
    parser.add_argument('--global_accuracy', default=0.0)
    parser.add_argument('--global_f', default=20)
    parser.add_argument('--accuracy', default=0.9)
    parser.add_argument('--log_path', default=LOG_PATH)
    parser.add_argument('--result_path', default=RESULT_PATH)
    parser.add_argument('--parallelism_candidates', default=5)
    parser.add_argument('--parallelism_exports', default=5)
    args = parser.parse_args()
    
    log_path = args.log_path
    result_path = args.result_path
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    latency_dict = {}
    
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    ## Read our query image
    latency_dict["image_path"] = args.image_path
    img = Image.open(args.image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.resize((940, 560))
    
    ## Initialize embedding model
    print("Initializing model")
    model, device = get_model()
    for _ in range(5):
        syntethic_img = np.random.rand(560, 940, 3).astype(np.float32)
        _ = inference(model, syntethic_img, device) # Warmup
    start = time.time()
    embeds = inference(model, np.array(img), device)  # Get embeddings
    inference_time = time.time()
    latency_dict["inference_ms"] = (inference_time - start)*1000

    ## Search global index
    candidates, global_search_times = search_global("global", embeds.detach().numpy(), ["collection"], int(args.global_k), float(args.global_accuracy), args.global_f)
    global_search = time.time()
    latency_dict["search_global_ms"] = (global_search - inference_time)*1000
    
    print("Candidates found:")
    print(candidates)
    latency_dict["candidates"] = candidates

    ## Perform queries
    print("Performing search")
    fragments = []
    latency_dict["frame_search_retrieve"] = []
    gb_retrieved_total = 0
    output_fields=["offset", "pk"]
    start_intra_search = time.time()
    with ThreadPoolExecutor(max_workers=int(args.parallelism_candidates)) as executor:
        search_partial = partial(
            search, 
            milvus_client=client, 
            embedding=embeds.detach().numpy(), 
            fields=output_fields, 
            local_k=int(args.local_k),
            fragment_offset=int(args.fragment_offset), 
            accuracy=float(args.accuracy), 
            result_path=result_path,
            parallelism=int(args.parallelism_exports)
        )
        futures = {executor.submit(search_partial, collection_name=collection_name): collection_name for collection_name in candidates}

        for future in as_completed(futures):
            collection_name = futures[future]
            try:
                fragment, gb_retrieved, log = future.result()
                fragments.extend(fragment)
                gb_retrieved_total += gb_retrieved
                latency_dict["frame_search_retrieve"].append(log)
            except Exception as e:
                print(f"Error processing collection {collection_name}: {e}")
    end_intra_search = time.time()
    ## Log
    latency_dict["intra_search_ms"] = (end_intra_search - start_intra_search)*1000
    latency_dict["frame_count"] = process_files(fragments, result_path)
    latency_dict["total_gb_retrieved"] = gb_retrieved_total
    latency_dict["search_global_times"] = global_search_times
    config = {
        "image_path": args.image_path,
        "global_k": args.global_k,
        "local_k": args.local_k,
        "fragment_offset": args.fragment_offset,
        "global_accuracy": args.global_accuracy,
        "accuracy": args.accuracy,
        "log_path": log_path,
        "result_path": result_path,
        "parallelism_candidates": args.parallelism_candidates,
        "parallelism_exports": args.parallelism_exports,
    }
    latency_dict["config"] = config
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{log_path}/query_logs_{timestamp}.json", "w") as f:
        json.dump(latency_dict, f)

    
if __name__ == "__main__":
    main()
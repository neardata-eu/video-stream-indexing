"""
In order to ensure correct execution of this script, please move it to '/streamsense/query/'
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pymilvus import (
    connections,
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

from index_utils import search_global


def main():
    parser = argparse.ArgumentParser(description='Milvus Query Demo')
    parser.add_argument('--image_path', default='/project/benchmarks/experiment3/cholec_frame_ref.png')
    parser.add_argument('--global_k', default=5)
    parser.add_argument('--global_accuracy', default=0.0)
    parser.add_argument('--global_f', default=20)
    parser.add_argument('--log_path', default=LOG_PATH)
    parser.add_argument('--result_path', default=RESULT_PATH)
    args = parser.parse_args()
    
    log_path = args.log_path
    result_path = args.result_path
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    latency_dict = {}
    
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)

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
    candidates, search_times = search_global("global", embeds.detach().numpy(), ["collection"], int(args.global_k), float(args.global_accuracy), args.global_f)
    global_search = time.time()
    latency_dict["search_global_ms"] = (global_search - inference_time)*1000
    
    print("Candidates found:")
    print(candidates)
    latency_dict["candidates"] = candidates
    
    config = {
        "image_path": args.image_path,
        "global_k": args.global_k,
        "global_accuracy": args.global_accuracy,
        "log_path": log_path,
        "result_path": result_path,
        "search_times": search_times,
    }
    latency_dict["config"] = config
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{log_path}/query_logs_{timestamp}.json", "w") as f:
        json.dump(latency_dict, f)

    
if __name__ == "__main__":
    main()
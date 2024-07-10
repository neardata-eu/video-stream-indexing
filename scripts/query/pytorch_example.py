import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pymilvus import (
    connections,
    MilvusClient
)

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from policies.constants import (MILVUS_HOST, MILVUS_PORT, MILVUS_NAMESPACE,
                                LOG_PATH, RESULT_PATH)
from policies.components import get_model, inference

from index_utils import search, search_global


class VectorDataset(Dataset):
    def __init__(self, milvus, embedding, global_k, global_accuracy, global_f, local_k, fragment_offset, accuracy, result_path, frame_path, parallelism_candidates, parallelism_exports):
        # Search Global Index for candidates
        candidates, _ = search_global("global", embedding.detach().numpy(), ["collection"], int(global_k), float(global_accuracy), global_f)
        
        # Generate video fragments
        fragments = []
        with ThreadPoolExecutor(max_workers=int(parallelism_candidates)) as executor:
            search_partial = partial(
                search, 
                milvus_client=milvus, 
                embedding=embedding.detach().numpy(), 
                fields=["offset", "pk"], 
                local_k=int(local_k),
                fragment_offset=int(fragment_offset), 
                accuracy=float(accuracy), 
                result_path=result_path,
                parallelism=int(parallelism_exports)
            )
            futures = {executor.submit(search_partial, collection_name=collection_name): collection_name for collection_name in candidates}

            for future in as_completed(futures):
                collection_name = futures[future]
                try:
                    fragment, _, _ = future.result()
                    fragments.extend(fragment)
                except Exception as e:
                    print(f"Error processing collection {collection_name}: {e}")

        # Save individual images on disk
        self.frame_path = frame_path # Path to store the frame .jpgs
        frame_count = 0
        print("Generating dataset...")
        for file in fragments:
            cap = cv2.VideoCapture(f"{result_path}/{file}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(f"{frame_path}/frame_{frame_count}.jpg", frame)
                frame_count += 1
            cap.release()
        self.frame_count = frame_count # Dataset length
            
    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        image = Image.open(f"{self.frame_path}/frame_{idx}.jpg")
        image_array = np.array(image)
        return image_array


def main():
    parser = argparse.ArgumentParser(description='PyTorch Query Demo')
    parser.add_argument('--image_path', default='/project/benchmarks/experiment3/cholec_frame_ref.png')
    parser.add_argument('--global_k', default=3)
    parser.add_argument('--local_k', default=20)
    parser.add_argument('--fragment_offset', default=5)
    parser.add_argument('--global_accuracy', default=0.0)
    parser.add_argument('--global_f', default=20)
    parser.add_argument('--accuracy', default=0.9)
    parser.add_argument('--log_path', default=LOG_PATH)
    parser.add_argument('--result_path', default=RESULT_PATH)
    parser.add_argument('--frame_path', default=f"{RESULT_PATH}/frames")
    parser.add_argument('--parallelism_candidates', default=3)
    parser.add_argument('--parallelism_exports', default=3)
    args = parser.parse_args()
    
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    frame_path = args.frame_path
    os.makedirs(frame_path, exist_ok=True)
    
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    
    ## Read image and perform inference
    model, device = get_model()
    img = Image.open("/project/benchmarks/experiment3/cholec_frame_ref.png")
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.resize((940, 560))
    embedding = inference(model, img, device)

    ## Create the dataset and dataloader
    dataset = VectorDataset(
        milvus=client,
        embedding=embedding,
        global_k=args.global_k,
        global_accuracy=args.global_accuracy,
        global_f=args.global_f,
        local_k=args.local_k,
        fragment_offset=args.fragment_offset,
        accuracy=args.accuracy,
        result_path=args.result_path,
        parallelism_candidates=args.parallelism_candidates,
        parallelism_exports=args.parallelism_exports,
        frame_path=frame_path
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    ## Example use
    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx} with shape: {batch.shape}")
        last = batch
        
    # Display last batch in separate window
    for idx, img in enumerate(last):
        # Save to disk
        img = Image.fromarray(np.array(img, dtype=np.uint8))
        img.save(f"{result_path}/last_frame_{idx}.jpg")
    
        
if __name__ == "__main__":
    main()
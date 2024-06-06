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
import torchvision
import torch
from torchvision import transforms
from torch import nn
import cv2

import subprocess
import time
from collections import defaultdict
import json
from datetime import datetime

from policies.constants import (MILVUS_HOST, MILVUS_PORT, MILVUS_NAMESPACE)


latency_dict = {}


def inference(model, image, device):
    """Extract features from an image"""
    image = cv2.resize(np.array(image), (384, 216))
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_batch = preprocess(image).unsqueeze(0)
    embedding = model(input_batch.to(device))
    return embedding.cpu()


class FeatureResNet(nn.Module):
    """ResNet model for feature extraction."""
    def __init__(self, num_features = 4096):
        super(FeatureResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def search_global(collection_name, embedding, fields, k=4):
    collection = Collection(collection_name)
    collection.load()
    
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=k*30, output_fields=fields)
    
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


def search(milvus_client, collection_name, embedding, fields, k=1):
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
            if (hit.distance < 0.9):
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
                
                gb_retrieved += os.path.getsize(f"/project/results/{collection_name}_{hit.pk}.h264") / (1024 ** 3)
    return hit_num, fail_num, gb_retrieved


def main():   
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)
    client = MilvusClient()

    ## Read our query image
    image_path = "example_frame.png"
    img = Image.open(image_path)
    
    ## Initialize embedding model
    print("Initializing model")
    model = FeatureResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    start = time.time()
    embeds = inference(model, np.array(img), device)  # Get embeddings
    inference_time = time.time()
    latency_dict["inference_ms"] = (inference_time - start)*1000

    ## Search global index
    candidates = search_global("global", embeds.detach().numpy(), ["collection"])
    global_search = time.time()
    latency_dict["search_global_ms"] = (global_search - inference_time)*1000
    
    print("Candidates found:")
    print(candidates)

    ## Perform queries
    print("Performing search")
    hit_num = 0
    fail_num = 0
    gb_retrieved = 0
    
    #collection_list = utility.list_collections()
    latency_dict["frame_search_retrieve"] = []
    for collection_name in candidates: # Search in the candidate collections
        output_fields=["offset", "pk"]
        hit, fail, gb = search(client, collection_name, embeds.detach().numpy(), output_fields)
        hit_num += hit
        fail_num += fail
        gb_retrieved += gb

    print(f"Number of coincidences found in the database: {hit_num}/{hit_num+fail_num}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"/project/results/query_logs_{timestamp}.json", "w") as f:
        json.dump(latency_dict, f)

    
if __name__ == "__main__":
    main()
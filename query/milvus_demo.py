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

import os
import subprocess
import time


def inference(model, image):
    """Extract features from an image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.resize(np.array(image), (384, 216))
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_batch = preprocess(image).unsqueeze(0)
    embedding = model(input_batch.to(device))
    return embedding.cpu().detach().numpy()


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
            videos.append(hit.collection)
            
    return list(set(videos))[:k]


def search(milvus_client, collection_name, embedding, fields, k=1):
    collection = Collection(collection_name)
    collection.load()
    
    result = collection.search(embedding, "embeddings", {"metric_type": "COSINE"}, limit=k, output_fields=fields)
    
    hit_num = 0
    fail_num = 0
    gb_retrieved = 0
    for hits in result:
        for hit in hits:
            if (hit.distance < 0.9):
                fail_num += 1
            else:
                print(f"Processing hit {hit.pk} with distance {hit.distance}")
                bounds = milvus_client.get(collection_name=collection_name, ids=[int(hit.pk)-20, int(hit.pk)+20], output_fields=["offset"])
                
                os.environ['BEGIN_OFFSET'] = bounds[0]["offset"]
                os.environ['END_OFFSET'] = bounds[1]["offset"]
                env = os.environ.copy()
                
                subprocess.run(['bash', '/project/scripts/export.sh', collection_name, f"{collection_name}_{hit.pk}"], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                hit_num += 1
                
                gb_retrieved += os.path.getsize(f"/project/results/{collection_name}_{hit.pk}.h264") / (1024 ** 3)
    return hit_num, fail_num, gb_retrieved


def main():   
    ## Connect to Milvus
    print("Connecting to Milvus")
    connections.connect("default", host="localhost", port="19530")
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
    embeds = inference(model, np.array(img))  # Get embeddings

    ## Search global index
    candidates = search_global("global", embeds, ["collection"])
    
    print("Candidates found:")
    print(candidates)

    ## Perform queries
    print("Performing search")
    hit_num = 0
    fail_num = 0
    gb_retrieved = 0
    
    #collection_list = utility.list_collections()
    for collection_name in candidates: # Search in the candidate collections
        output_fields=["offset", "pk"]
        hit, fail, gb = search(client, collection_name, embeds, output_fields)
        hit_num += hit
        fail_num += fail
        gb_retrieved += gb
                        
    print(f"Number of coincidences found in the database: {hit_num}/{hit_num+fail_num}")

    
if __name__ == "__main__":
    main()
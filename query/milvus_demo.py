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


def inference(model, image):
    """Extract features from an image"""
    image = cv2.resize(np.array(image), (384, 216))
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_batch = preprocess(image).unsqueeze(0)
    embedding = model(input_batch)
    return embedding


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

    ## Perform search
    print("Performing search")
    search_params = {
        "metric_type": "COSINE"
    }
    results = {}
    collection_list = utility.list_collections()
    for collection_name in collection_list:
        collection = Collection(collection_name)
        collection.load()
        result = collection.search(embeds.detach().numpy(), "embeddings", search_params, limit=1, output_fields=["offset", "pk"])
        results[collection_name] = result
    
    ## Get Pravega video segment
    print("Getting video segments from Pravega")
    script_path = '/project/scripts/export.sh'
    env = os.environ.copy()
    hit_num = 0
    fail_num = 0
    for key in results.keys():
        initial_offset = 0
        for hits in results[key]:
            for hit in hits:
                if (hit.distance < 0.85):
                    fail_num += 1
                else:
                    hit_num += 1
                    print(f"Processing hit {hit.pk} with distance {hit.distance}")
                    bounds = client.get(collection_name=key, ids=[int(hit.pk)-10, int(hit.pk)+10], output_fields=["offset"])
                    os.environ['BEGIN_OFFSET'] = bounds[0]["offset"]
                    os.environ['END_OFFSET'] = bounds[1]["offset"]
                    env = os.environ.copy()
                    subprocess.run(['bash', script_path, key, f"{key}_{hit.pk}"], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
    print(f"Number of coincidences found in the database: {hit_num}/{hit_num+fail_num}")

    
if __name__ == "__main__":
    main()
from policies.model import FeatureResNet

from torchvision import transforms
import numpy as np
import torch
import cv2

def get_model():
    """Load the model and move it to the device"""
    model = FeatureResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

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

def do_sampling():
    """Determine the sampling method for the global index"""
    counter = 0

    def time_based_sampling():
        nonlocal counter
        counter += 1
        return (counter-1) % 30 == 0

    return time_based_sampling
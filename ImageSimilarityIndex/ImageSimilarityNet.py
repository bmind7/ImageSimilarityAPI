import os
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

WEIGHTS_FILE = "resnet50-11ad3fa6.pth"


class ImageSimilarityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.model = resnet50()
        full_path = os.path.join("ImageSimilarityIndex", WEIGHTS_FILE)
        self.model.load_state_dict(torch.load(full_path))
        self.model = create_feature_extractor(
            self.model, return_nodes=['flatten'])
        self.model.eval()

    def calculate(self, img1, img2):
        with torch.no_grad():
            batch = torch.stack(
                (self.preprocess(img1), self.preprocess(img2)))
            result = self.model(batch)['flatten']
            return torch.dot(
                torch.nn.functional.normalize(result[0], p=2.0, dim=0),
                torch.nn.functional.normalize(result[1], p=2.0, dim=0)).item()


model = ImageSimilarityNet()

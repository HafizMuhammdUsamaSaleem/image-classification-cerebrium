import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class PytorchModel:
    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.preprocess = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

    def preprocess_numpy(self, img_np: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(img_np).convert("RGB")
        return self.preprocess(img).unsqueeze(0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(tensor)

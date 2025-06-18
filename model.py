# model.py

import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class Preprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def process(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).numpy().astype(np.float32)
        return tensor

class ONNXModel:
    def __init__(self, model_path="model.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_tensor: np.ndarray) -> int:
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return int(np.argmax(outputs[0], axis=1)[0])

import torch
from pytorch_model import PytorchModel
from PIL import Image
import numpy as np

model = PytorchModel()
ckpt = torch.load("weights/pytorch_model_weights.pth", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = model.preprocess_numpy(np.array(img))
    out = model(inp)
    print(f"{img_path} → PyTorch pred: {int(out.argmax())}")

predict("images/n01440764_tench.jpeg")
predict("images/n01667114_mud_turtle.JPEG")

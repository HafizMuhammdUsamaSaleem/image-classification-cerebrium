import torch
from torchvision import models

def convert_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print("✅ ONNX model created successfully from pretrained weights.")

if __name__ == "__main__":
    convert_model()

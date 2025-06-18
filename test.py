# test.py

from model import Preprocessor, ONNXModel

def main():
    pre = Preprocessor()
    model = ONNXModel("model.onnx")

    test_images = [
        "images/n01440764_tench.jpeg",
        "images/n01667114_mud_turtle.JPEG"
    ]

    for img_path in test_images:
        tensor = pre.process(img_path)
        pred = model.predict(tensor)
        print(f"{img_path} → Predicted class ID: {pred}")

if __name__ == "__main__":
    main()

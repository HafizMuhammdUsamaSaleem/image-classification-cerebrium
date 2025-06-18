# app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from model import Preprocessor, ONNXModel
import numpy as np
import io
from PIL import Image

app = FastAPI()
pre = Preprocessor()
model = ONNXModel("model.onnx")
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    tensor = pre.transform(img).unsqueeze(0).numpy().astype(np.float32)
    prediction = model.predict(tensor)
    return {"predicted_class_id": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)

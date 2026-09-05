"""FastAPI backend for the Manufacturing Defect Classification web UI."""

import os
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from inference_pipeline import InferencePipeline


app = FastAPI(title="Manufacturing Defect Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)

print("Loading inference pipeline...")
pipeline = InferencePipeline()
print(f"Loaded. Available domains: {list(pipeline.specialist_models.keys())}")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>MDC API is running</h1><p>Upload an image to /predict</p>"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    temp_path = f"outputs/uploaded_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        gradcam_path = "outputs/gradcam_api.png"
        result = pipeline.predict(temp_path, save_gradcam_path=gradcam_path)

        gradcam_b64 = None
        if os.path.exists(gradcam_path):
            with open(gradcam_path, "rb") as f:
                gradcam_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = {
            "domain": result["domain"],
            "domain_confidence": round(result["domain_confidence"], 4),
            "defect": result["defect"],
            "defect_confidence": round(result["defect_confidence"], 4),
            "domain_probabilities": {k: round(v, 4) for k, v in result["domain_probabilities"].items()},
            "defect_probabilities": {k: round(v, 4) for k, v in result["defect_probabilities"].items()},
            "gradcam_base64": gradcam_b64,
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "loaded_domains": list(pipeline.specialist_models.keys())}

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import logging
import tempfile
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docling")

app = FastAPI(title="Docling PDF Converter")

_converter = None

def get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        logger.info(f"Initializing Docling (PyTorch: {torch.__version__})")
        _converter = DocumentConverter()
    return _converter

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": False, "service": "docling"}

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return JSONResponse(400, {"error": "Only PDF files allowed"})
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        converter = get_converter()
        result = converter.convert(tmp_path)
        
        # Simple markdown output
        markdown = result.document.export_to_markdown()
        
        return {"markdown": markdown, "filename": file.filename}
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return JSONResponse(500, {"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

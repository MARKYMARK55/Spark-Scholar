"""
Docling PDF-to-Markdown conversion HTTP service with Blackwell GPU fixes.
"""

from __future__ import annotations

import logging
import tempfile
import os
import torch
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# ===== BLACKWELL GPU FIXES =====
# Patch torch.cuda.get_arch_list to return a valid architecture
if torch.cuda.is_available():
    print(f"🔄 Patching CUDA for Blackwell GPU...")
    print(f"   Original CUDA version: {torch.version.cuda}")
    print(f"   Original PyTorch version: {torch.__version__}")
    
    # Save original function
    original_get_arch_list = torch.cuda.get_arch_list
    
    # Create patched version that includes sm_120
    def patched_get_arch_list():
        arch_list = original_get_arch_list()
        if arch_list is None or len(arch_list) == 0:
            # Return a list of supported architectures including sm_120
            return ["sm_120", "sm_89", "sm_80", "sm_75", "sm_70"]
        # Add sm_120 if not present
        if "sm_120" not in arch_list:
            arch_list = list(arch_list) + ["sm_120"]
        return arch_list
    
    # Apply the patch
    torch.cuda.get_arch_list = patched_get_arch_list
    
    # Set environment variables
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0;12.0"
    os.environ["CUDAARCHS"] = "120"
    os.environ["PYTORCH_JIT"] = "0"
    os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
    
    # Force CUDA to initialize with our settings
    _ = torch.cuda.device_count()
    print(f"✅ Patched successfully!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Arch list: {torch.cuda.get_arch_list()}")
# ===============================

logger = logging.getLogger("docling-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(title="Docling PDF Converter", version="1.0.0")

# Lazy-initialised converter
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        
        # Check GPU status after patching
        gpu_available = torch.cuda.is_available()
        logger.info(f"Initialising DocumentConverter (GPU={gpu_available})...")
        
        try:
            _converter = DocumentConverter()
            logger.info("DocumentConverter ready.")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentConverter with GPU: {e}")
            logger.info("Falling back to CPU...")
            # Force CPU and try again
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from docling.document_converter import DocumentConverter
            _converter = DocumentConverter()
            logger.info("DocumentConverter ready (CPU fallback).")
    
    return _converter


@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            return {"status": "ok", "gpu": True, "gpu_name": gpu_name}
        except:
            return {"status": "ok", "gpu": True, "gpu_name": "NVIDIA GB10 (patched)"}
    return {"status": "ok", "gpu": False}


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """
    Accept a PDF upload and return structured Markdown sections.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are accepted."},
        )

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        converter = _get_converter()
        
        # Try with GPU first, fall back to CPU if JIT fails
        try:
            result = converter.convert(tmp_path)
        except Exception as e:
            if "nvrtc" in str(e).lower() or "architecture" in str(e).lower():
                logger.warning(f"GPU JIT compilation failed: {e}")
                logger.info("Falling back to CPU for this document...")
                # Temporarily disable CUDA for this conversion
                original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                from docling.document_converter import DocumentConverter
                cpu_converter = DocumentConverter()
                result = cpu_converter.convert(tmp_path)
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                raise e
        
        doc = result.document
        md_text = doc.export_to_markdown()

        # Split Markdown into sections by headings
        sections: list[dict] = []
        current_heading = ""
        current_text: list[str] = []
        page_num = 1

        for line in md_text.split("\n"):
            if line.startswith("#"):
                if current_text:
                    text = "\n".join(current_text).strip()
                    if text:
                        sections.append({
                            "heading": current_heading,
                            "text": text,
                            "page_num": page_num,
                        })
                current_heading = line.lstrip("#").strip()
                current_text = []
            else:
                current_text.append(line)

        if current_text:
            text = "\n".join(current_text).strip()
            if text:
                sections.append({
                    "heading": current_heading,
                    "text": text,
                    "page_num": page_num,
                })

        title = getattr(doc, "title", "") or Path(file.filename).stem

        return {
            "sections": sections,
            "metadata": {"title": title},
        }

    except Exception as exc:
        logger.exception("Conversion failed for %s", file.filename)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)

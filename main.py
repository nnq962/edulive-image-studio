import contextlib
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

# Import các hàm tiện ích
from flux_kontext import _prepare_pipeline, run_flux_edit
from utils.logger import LOGGER

# --- 1. GLOBAL VARIABLE (Biến toàn cục để lưu Model) ---
pipeline_instance = None

# --- 2. LIFESPAN: Load Model 1 lần khi Server bật ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    LOGGER.info("Server đang khởi động: Đang load Model FLUX vào RAM/VRAM...")
    
    try:
        # Gọi hàm prepare của bạn ĐÚNG 1 LẦN ở đây
        # Lúc này nó sẽ check VRAM và chọn chiến thuật (model_offload hay sequential)
        pipeline_instance = _prepare_pipeline()
        LOGGER.info("Model FLUX đã sẵn sàng! API bắt đầu nhận request.")
    except Exception as e:
        LOGGER.error(f"Lỗi nghiêm trọng khi load model: {e}")
        # Có thể không raise lỗi để server vẫn chạy, nhưng API sẽ trả 500
    
    yield  # Server chạy và nhận request ở đây...
    
    # Đoạn này chạy khi bạn tắt Server (Ctrl+C)
    LOGGER.info("Server đang tắt: Giải phóng bộ nhớ...")
    if pipeline_instance:
        del pipeline_instance
    torch.cuda.empty_cache()

# Khởi tạo FastAPI với lifespan
app = FastAPI(lifespan=lifespan)

# CORS để frontend (port 3000) có thể gọi trực tiếp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. DTO (Data Transfer Object) cho Request ---
class EditRequest(BaseModel):
    image_url: str
    prompt: str
    guidance_scale: float = 2.5
    num_inference_steps: int = 28
    seed: Optional[int] = None

# --- 4. API ENDPOINT (Chỉ gọi hàm tạo ảnh) ---
@app.post("/generate")
async def generate_image(req: EditRequest):
    global pipeline_instance
    
    # Kiểm tra xem model có sống không
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng, vui lòng thử lại sau.")

    try:
        LOGGER.info(f"Nhận request: {req.prompt}")

        # Dùng pipeline đã load sẵn + hàm tiện ích, KHÔNG khởi tạo lại pipeline
        result_image = run_flux_edit(
            pipe=pipeline_instance,
            image_input=req.image_url,
            prompt=req.prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
        )

        # Chuyển ảnh sang bytes (PNG) và trả về trực tiếp
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="image/png")

    except Exception as e:
        LOGGER.error(f"Lỗi xử lý: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_upload")
async def generate_image_upload(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    guidance_scale: float = Form(2.5),
    num_inference_steps: int = Form(28),
    seed: Optional[int] = Form(None),
):
    """
    Endpoint nhận ảnh upload (multipart/form-data) + prompt và trả về ảnh đã chỉnh.
    """
    global pipeline_instance

    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng, vui lòng thử lại sau.")

    try:
        LOGGER.info(f"Nhận request upload: {prompt} | filename={file.filename}")

        # Đọc file upload vào RAM và convert sang PIL.Image
        file_bytes = await file.read()
        input_image = Image.open(BytesIO(file_bytes)).convert("RGB")

        # Chạy model với ảnh upload
        result_image = run_flux_edit(
            pipe=pipeline_instance,
            image_input=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="image/png")

    except Exception as e:
        LOGGER.error(f"Lỗi xử lý upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
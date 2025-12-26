import gc
import os
import subprocess
from typing import Optional, Union

import torch
from PIL import Image  # Cần import thêm Image để dùng thuật toán resize xịn (LANCZOS)
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from dotenv import load_dotenv

from utils.logger import LOGGER

# Lệnh này để tải các biến từ file .env vào chương trình
load_dotenv()

VRAM_THRESHOLD_MB = 26 * 1024  # ~26GB


def get_nvidia_smi_usage():
    try:
        # Gọi nvidia-smi để lấy thông số memory.used và memory.total
        # flags: --query-gpu giúp lấy đúng số liệu, --format=csv giúp dễ tách dữ liệu
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )

        # Kết quả trả về dạng: "25835, 46068"
        used, total = result.strip().split(",")

        return {
            "memory_used": int(used),
            "memory_total": int(total),
            "memory_free": int(total) - int(used),
        }

    except FileNotFoundError:
        LOGGER.error("Lỗi: Không tìm thấy lệnh nvidia-smi (bạn có đang chạy trên máy có driver NVIDIA không?)")
    except Exception as e:  # noqa: BLE001
        LOGGER.error(f"Lỗi khác: {e}")

    return None


# --- HÀM TIỆN ÍCH: TÍNH KÍCH THƯỚC TỐI ƯU ---
def get_optimal_size(width, height, target_pixels=1024 * 1024):
    """
    Tính toán kích thước mới sao cho:
    1. Tổng pixel xấp xỉ 1MP (để FLUX chạy ngon nhất)
    2. Giữ nguyên tỷ lệ khung hình (Aspect Ratio)
    3. Chiều dài/rộng chia hết cho 16
    """
    aspect_ratio = width / height

    # Tính chiều cao mới dựa trên diện tích mục tiêu
    new_h = int((target_pixels / aspect_ratio) ** 0.5)

    # Làm tròn cho chia hết cho 16
    new_h = round(new_h / 16) * 16

    # Tính chiều rộng tương ứng
    new_w = int(new_h * aspect_ratio)
    new_w = round(new_w / 16) * 16

    return new_w, new_h


def _prepare_pipeline():
    """
    Khởi tạo pipeline và chọn chế độ offload dựa trên dung lượng VRAM còn trống.
    - Nếu trống >= 26GB -> enable_model_cpu_offload
    - Ngược lại -> enable_sequential_cpu_offload
    """
    gc.collect()
    torch.cuda.empty_cache()

    usage = get_nvidia_smi_usage()
    has_enough_vram = usage is not None and usage["memory_free"] >= VRAM_THRESHOLD_MB

    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN"),
    )

    if has_enough_vram:
        LOGGER.info(
            f"VRAM trống {usage['memory_free']} MB đủ ngưỡng {VRAM_THRESHOLD_MB} MB -> dùng enable_model_cpu_offload"
        )
        pipe.enable_model_cpu_offload()
    else:
        free_str = usage['memory_free'] if usage else "unknown"
        LOGGER.info(f"VRAM trống {free_str} MB không đủ ngưỡng {VRAM_THRESHOLD_MB} MB -> dùng enable_sequential_cpu_offload")
        pipe.enable_sequential_cpu_offload()

    return pipe


def run_flux_edit(
    pipe: FluxKontextPipeline,
    image_input: Union[str, Image.Image],
    prompt: str,
    guidance_scale: float = 2.5,
    num_inference_steps: int = 28,
    target_pixels: int = 1024 * 1024,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Chạy edit ảnh với pipeline đã được khởi tạo sẵn (KHÔNG khởi tạo lại).
    - image_input:
        + str: local path hoặc URL (load_image hỗ trợ cả hai)
        + PIL.Image.Image: ảnh đã được load sẵn (ví dụ từ upload file)
    - prompt: văn bản hướng dẫn.
    Trả về đối tượng PIL.Image đã xử lý.
    """
    if isinstance(image_input, str):
        input_image = load_image(image_input)
    elif isinstance(image_input, Image.Image):
        input_image = image_input
    else:
        raise ValueError("image_input phải là đường dẫn/URL (str) hoặc PIL.Image.Image")

    # --- XỬ LÝ KÍCH THƯỚC THÔNG MINH ---
    original_w, original_h = input_image.size
    run_w, run_h = get_optimal_size(original_w, original_h, target_pixels=target_pixels)

    LOGGER.info(f"Kích thước gốc: {original_w}x{original_h} -> Kích thước chạy Model: {run_w}x{run_h}")
    input_image = input_image.resize((run_w, run_h), Image.LANCZOS)

    generator = torch.manual_seed(seed) if seed is not None else None

    LOGGER.info("Đang xử lý...")
    image = pipe(
        prompt=prompt,
        image=input_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=run_w,
        height=run_h,
        generator=generator,
    ).images[0]

    # --- HẬU XỬ LÝ: trả ảnh về kích thước gốc ---
    if image.size != (original_w, original_h):
        LOGGER.info(f"Resize kết quả từ {image.size} về lại gốc {original_w}x{original_h}")
        image = image.resize((original_w, original_h), Image.LANCZOS)

    return image


def generate_flux_image(
    image_path: str,
    prompt: str,
    output_path: str = "output.png",
    guidance_scale: float = 2.5,
    num_inference_steps: int = 28,
    target_pixels: int = 1024 * 1024,
    seed: Optional[int] = None,
) -> str:
    """
    Hàm tiện lợi khi chạy script đơn lẻ (KHÔNG dùng cho API).
    Tự khởi tạo pipeline bên trong rồi lưu ảnh ra ổ cứng.
    """
    pipe = _prepare_pipeline()
    image = run_flux_edit(
        pipe=pipe,
        image_path=image_path,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        target_pixels=target_pixels,
        seed=seed,
    )

    image.save(output_path)
    LOGGER.info(f"Đã lưu ảnh ra {output_path}")
    return output_path


if __name__ == "__main__":
    # Ví dụ nhanh khi chạy trực tiếp file này:
    # generate_flux_image("https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=1xw:0.74975xh;center,top&resize=1200:*", "rotate the dog to face forward, front view, sitting", "output/remove_dog.png")
    pass
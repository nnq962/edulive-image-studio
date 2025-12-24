import torch
import gc
from PIL import Image # Cần import thêm Image để dùng thuật toán resize xịn (LANCZOS)
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os
from dotenv import load_dotenv

# Lệnh này để tải các biến từ file .env vào chương trình
load_dotenv()

# --- HÀM TIỆN ÍCH: TÍNH KÍCH THƯỚC TỐI ƯU ---
def get_optimal_size(width, height, target_pixels=1024*1024):
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

# 1. Dọn dẹp bộ nhớ trước
gc.collect()
torch.cuda.empty_cache()

# 2. Load Model
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    token = os.getenv("HF_TOKEN")
)
# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

# 3. Load ảnh input
image_url = "/home/quyetnn/nnq962/dev/flux/images.webp" 
# Hoặc thay bằng đường dẫn ảnh local của bạn:
# image_url = "/home/quyetnn/nnq962/dev/flux/anh_goc.jpg"

input_image = load_image(image_url)

# --- XỬ LÝ KÍCH THƯỚC THÔNG MINH ---
# Lưu lại kích thước gốc để sau này trả hàng cho đúng
original_w, original_h = input_image.size

# Tính kích thước tối ưu cho Model (xấp xỉ 1MP)
run_w, run_h = get_optimal_size(original_w, original_h)

print(f"Kích thước gốc: {original_w}x{original_h} -> Kích thước chạy Model: {run_w}x{run_h}")

# Resize ảnh input về kích thước tối ưu (Dùng LANCZOS để ảnh nét nhất có thể)
input_image = input_image.resize((run_w, run_h), Image.LANCZOS)

print("Đang xử lý...")
# 4. Chạy model
image = pipe(
    prompt="make the cat happy",
    image=input_image,
    guidance_scale=2.5,     
    num_inference_steps=28, 
    width=run_w,            # Chạy ở kích thước tối ưu
    height=run_h,           
    generator=torch.manual_seed(42) 
).images[0]

# --- HẬU XỬl LÝ (POST-PROCESSING) ---
# (Tuỳ chọn) Đưa ảnh kết quả về lại đúng kích thước gốc ban đầu của User
if image.size != (original_w, original_h):
    print(f"Resize kết quả từ {image.size} về lại gốc {original_w}x{original_h}")
    image = image.resize((original_w, original_h), Image.LANCZOS)

image.save("happy_cat.png")

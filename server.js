require('dotenv').config();
const express = require('express');
const { Midjourney } = require("midjourney");
const cors = require('cors');
const path = require('path');

const app = express();
app.use(express.json()); // Để đọc được JSON body
app.use(cors());

app.use(express.static(path.join(__dirname, 'public')));

// --- CẤU HÌNH (Thay bằng thông số thật của bạn) ---
const CONFIG = {
    ServerId: process.env.MJ_SERVER_ID,
    ChannelId: process.env.MJ_CHANNEL_ID,
    SalaiToken: process.env.MJ_SALAI_TOKEN,
    Debug: true,
    Ws: true,
};

// Khởi tạo Client Midjourney
const client = new Midjourney(CONFIG);

// --- HÀM XỬ LÝ LOGIC (The Chef 🧑‍🍳) ---
// Nhiệm vụ: Biến cục JSON đẹp đẽ của bạn thành chuỗi lệnh --ar --v xấu xí
function generateMidjourneyPrompt(body) {
    let promptString = body.prompt; // Lấy prompt gốc trước

    // 1. Xử lý Tỷ lệ (Aspect Ratio)
    if (body.aspect_ratio) {
        promptString += ` --ar ${body.aspect_ratio}`;
    }

    // 2. Xử lý Model
    // Logic mới: "anime" -> niji 6, còn lại mặc định là v7
    if (body.model === 'anime') {
        promptString += ` --niji 6`;
    } else if (body.model === 'v6') {
        // Giữ lại một cửa lùi nếu user thích dùng bản cũ
        promptString += ` --v 6.1`; 
    } else {
        // Mặc định ("standard" hoặc không điền gì) sẽ là V7
        promptString += ` --v 7`;
    }

    // 3. Xử lý Stylize (Độ nghệ thuật)
    switch (body.stylize) {
        case 'low':
            promptString += ` --s 50`; break;
        case 'high':
            promptString += ` --s 750`; break;
        case 'very_high':
            promptString += ` --s 1000`; break;
        case 'medium':
        default:
            promptString += ` --s 100`; // Mặc định là Medium
            break;
    }

    // 4. Xử lý Negative Prompt (Cái không muốn vẽ)
    if (body.negative_prompt) {
        promptString += ` --no ${body.negative_prompt}`;
    }

    return promptString;
}

// --- API 1: GENERATE (Vẽ ảnh) ---
app.post('/api/v1/generate', async (req, res) => {
    try {
        const { prompt } = req.body;
        
        // Validate cơ bản
        if (!prompt) {
            return res.status(400).json({ error: "Thiếu 'prompt' rồi sếp ơi!" });
        }

        // Bước 1: Xào nấu prompt
        const finalPrompt = generateMidjourneyPrompt(req.body);
        console.log(">>> Đang gửi lệnh cho Bot:", finalPrompt);

        // Bước 2: Gọi Bot vẽ (Imagine)
        // Lưu ý: Hàm này sẽ chờ cho đến khi vẽ xong 4 ảnh (Grid)
        const msg = await client.Imagine(finalPrompt, (uri, progress) => {
            console.log(`Tiến độ: ${progress}`);
        });

        if (!msg) {
            throw new Error("Bot không trả về kết quả (Có thể do lỗi mạng hoặc Token)");
        }

        // Bước 3: Trả kết quả về cho Client
        res.json({
            status: "success",
            data: {
                message_id: msg.id,       // QUAN TRỌNG: Dùng ID này để gọi API Upscale sau này
                grid_image_url: msg.uri,  // Link ảnh 4-trong-1
                content: msg.content,     // Nội dung prompt thực tế Bot nhận
                flags: msg.flags,          // Các cờ kỹ thuật
                hash: msg.hash // <--- THÊM DÒNG NÀY QUAN TRỌNG
            }
        });

    } catch (error) {
        console.error("Lỗi rồi:", error);
        res.status(500).json({ error: "Lỗi Server hoặc Discord", details: error.message });
    }
});

// --- API 2: UPSCALE (Tách ảnh) ---
app.post('/api/v1/upscale', async (req, res) => {
    try {
        // Nhận thêm tham số hash từ client gửi lên
        const { message_id, index, hash, flags } = req.body;

        // 1. Validate
        if (!message_id || !hash) { // Bắt buộc phải có hash
            return res.status(400).json({ error: "Thiếu 'message_id' hoặc 'hash' (Lấy từ API Generate)" });
        }
        if (!index || index < 1 || index > 4) {
            return res.status(400).json({ error: "Index phải từ 1 đến 4" });
        }

        console.log(`>>> Đang Upscale ảnh số ${index} của tin nhắn ${message_id}...`);

        // 2. Dùng hàm Upscale chuẩn của thư viện
        // Hàm này sẽ tự động tìm đúng nút U1/U2... dựa trên hash để bấm
        const msg = await client.Upscale({
            index: index,
            msgId: message_id,
            hash: hash,
            flags: flags || 0 // Nếu không có flags thì mặc định là 0
        });

        if (!msg) {
            throw new Error("Không thể Upscale (Có thể ảnh đã hết hạn hoặc tham số sai).");
        }

        console.log(">>> Upscale thành công:", msg.uri);

        // 3. Trả về kết quả
        res.json({
            status: "success",
            data: {
                original_message_id: message_id,
                upscaled_image_url: msg.uri,
                content: msg.content
            }
        });

    } catch (error) {
        console.error("Lỗi Upscale:", error);
        res.status(500).json({ error: "Lỗi Upscale", details: error.message });
    }
});


// --- KHỞI ĐỘNG SERVER ---
const PORT = process.env.PORT || 3000;
async function startServer() {
    try {
        await client.init();
        console.log(">>> Bot đã đăng nhập thành công!");
        
        app.listen(PORT, '0.0.0.0', () => { // Thêm '0.0.0.0' để mở full cổng mạng
            console.log(`>>> Server đang chạy!`);
            console.log(`>>> Link Local:   http://localhost:${PORT}`);
            console.log(`>>> Link cho AE:  http://192.168.1.144:${PORT}`);
        });
    } catch (err) {
        console.error("Lỗi:", err);
    }
}
startServer();

const { Midjourney } = require("midjourney");

// 1. Điền 3 thông số bạn vừa lấy vào đây
const SERVER_ID = "";
const CHANNEL_ID = "";
const TOKEN = "";

const client = new Midjourney({
  ServerId: SERVER_ID,
  ChannelId: CHANNEL_ID,
  SalaiToken: TOKEN,
  Debug: true,
  Ws: true, 
});

async function testConnection() {
  try {
    console.log("dang ket noi voi Discord...");
    
    // Khởi tạo kết nối
    await client.init();
    
    console.log(">>> KET NOI THANH CONG! Token van tot.");
    console.log(">>> Bot da san sang tai Server: " + SERVER_ID);
    
    // Nếu đã mua gói, thử vẽ 1 cái (bỏ comment dòng dưới để chạy)
    // const msg = await client.Imagine("a cute robot logo", (uri, progress) => {
    //   console.log("Tien do ve: " + progress);
    // });
    // console.log("Anh da ve xong:", msg.uri);

  } catch (error) {
    console.log(">>> LOI ROI: Token khong dung hoac mang co van de.");
    console.error(error);
  }
}

testConnection();
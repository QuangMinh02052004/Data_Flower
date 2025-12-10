# 🌸 Hệ thống Tìm Kiếm Hoa Bằng Hình Ảnh

Hệ thống AI tìm kiếm và nhận dạng hoa tự động từ ảnh, tích hợp với website Bloomie Flower Shop.

## ✨ Tính năng

- 🎯 **Nhận dạng loại hoa**: Tự động phát hiện loại hoa (Hồng, Cúc, Hướng dương, Tulip, Bồ công anh)
- 🎨 **Phân tích màu sắc**: Trích xuất màu sắc chủ đạo từ ảnh
- 📸 **Upload ảnh hoặc chụp trực tiếp**: Hỗ trợ cả upload file và chụp ảnh từ camera
- 🔍 **Filter thông minh**: Tự động filter sản phẩm theo loại hoa và màu sắc

## 🏗️ Kiến trúc

```
┌─────────────┐      HTTP POST       ┌──────────────┐
│  ASP.NET    │ ───────────────────> │  Python API  │
│  Core       │  (multipart/form)    │  (FastAPI)   │
│  Frontend   │ <─────────────────── │  Port 8000   │
└─────────────┘      JSON Response   └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │ TensorFlow   │
                                      │ Model (.keras)│
                                      └──────────────┘
```

## 📋 Yêu cầu hệ thống

### Python Requirements
- Python 3.9+
- TensorFlow 2.15+
- FastAPI
- Uvicorn
- scikit-learn
- Pillow

### ASP.NET Core Requirements
- .NET 8.0+
- HttpClientFactory (đã có sẵn)

## 🚀 Cài đặt và Chạy

### Bước 1: Setup Python Environment

```bash
cd /Users/lequangminh/Downloads/DACN-main/flower-image-search

# Tạo virtual environment (khuyến nghị)
python3 -m venv venv
source venv/bin/activate  # Trên macOS/Linux
# hoặc
venv\Scripts\activate     # Trên Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Khởi động Python API Service

```bash
# Đảm bảo bạn đang ở thư mục flower-image-search
python image_search_api.py
```

API sẽ chạy tại: **http://localhost:8000**

Kiểm tra API:
- Health check: http://localhost:8000/health
- Documentation: http://localhost:8000/docs

### Bước 3: Chạy ASP.NET Core Application

```bash
cd /Users/lequangminh/Downloads/DACN-main

# Chạy project
dotnet run
```

Website sẽ chạy tại: **http://localhost:5229**

## 📖 Sử dụng

### 1. Tìm kiếm bằng Upload Ảnh

1. Vào trang chủ hoặc trang sản phẩm
2. Click vào nút **Upload** (icon 📤) trong thanh tìm kiếm
3. Chọn ảnh hoa từ máy tính
4. Hệ thống sẽ tự động phân tích và redirect đến kết quả

### 2. Tìm kiếm bằng Camera

1. Vào trang chủ hoặc trang sản phẩm
2. Click vào nút **Camera** (icon 📷) trong thanh tìm kiếm
3. Cho phép truy cập camera
4. Chụp ảnh hoa
5. Click "Gửi ảnh" để phân tích

## 🔧 API Endpoints

### POST `/analyze`

Phân tích ảnh upload và trả về thông tin loại hoa, màu sắc.

**Request:**
```
Content-Type: multipart/form-data
file: <image file>
```

**Response:**
```json
{
  "success": true,
  "flower_type": "Hồng",
  "confidence": 95.5,
  "dominant_colors": ["Đỏ", "Hồng"],
  "message": "Phát hiện hoa Hồng với độ tin cậy 95.5%"
}
```

## 🎓 Model Training

Model được training với:
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Dataset**: 5 loại hoa (daisy, dandelion, rose, sunflower, tulip)
- **Input size**: 224x224x3
- **Optimizer**: Adam
- **Training**: 2 phases (freeze base → fine-tuning)

### Re-train Model (nếu cần)

```bash
# Train từ đầu
python train_model_optimized.py

# Extract features cho search
python extract_features.py
```

## 🐛 Troubleshooting

### Lỗi: "Không thể kết nối đến dịch vụ phân tích ảnh"

**Nguyên nhân**: Python API chưa chạy hoặc chạy sai port.

**Giải pháp**:
1. Kiểm tra Python API đang chạy: `curl http://localhost:8000/health`
2. Khởi động lại Python API: `python image_search_api.py`
3. Kiểm tra port trong ProductController.cs (dòng 17): `PYTHON_API_URL = "http://localhost:8000/analyze"`

### Lỗi: "Module not found"

**Giải pháp**:
```bash
pip install -r requirements.txt
```

### Lỗi: Camera không hoạt động

**Giải pháp**:
1. Đảm bảo bạn đang truy cập qua HTTPS hoặc localhost
2. Cho phép quyền truy cập camera trong browser settings
3. Thử browser khác (Chrome/Edge khuyến nghị)

### Model file không tìm thấy

**Giải pháp**:
```bash
cd flower-image-search
ls -la *.keras  # Kiểm tra xem có file .keras không

# Nếu không có, cần train lại:
python train_model_optimized.py
```

## 📊 Performance

- **Inference time**: ~100-300ms/ảnh
- **Accuracy**: ~85-95% (tùy loại hoa và chất lượng ảnh)
- **Supported formats**: JPG, PNG, WebP
- **Max file size**: 5MB

## 🔐 Security Notes

- API chạy trên localhost, chỉ accept requests từ local
- Trong production, nên:
  - Cấu hình CORS cụ thể
  - Thêm authentication
  - Rate limiting
  - Input validation

## 📝 Mapping Database

Model nhận dạng các loại hoa sau:

| Model Class | Tên tiếng Việt | Tìm trong FlowerType |
|-------------|----------------|----------------------|
| rose        | Hồng           | Hoa Hồng             |
| daisy       | Cúc            | Hoa Cúc              |
| sunflower   | Hướng dương    | Hoa Hướng Dương      |
| tulip       | Tulip          | Hoa Tulip            |
| dandelion   | Bồ công anh    | Hoa Bồ Công Anh      |

Màu sắc được mapping sang FlowerVariant.Color trong database.

## 🤝 Support

Nếu gặp vấn đề, kiểm tra:
1. Python API logs (terminal running `image_search_api.py`)
2. ASP.NET logs (terminal running `dotnet run`)
3. Browser console (F12)

## 📄 License

Private project for Bloomie Flower Shop.

---

**Phiên bản**: 1.0.0
**Cập nhật**: December 2025
**Tác giả**: Bloomie Development Team

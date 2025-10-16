# extract_features.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from tqdm import tqdm  # Thư viện để hiển thị progress bar, cài bằng: pip install tqdm

# 1. Tải mô hình đã huấn luyện
model_path = "flower_classifier_final.keras"
base_model = tf.keras.models.load_model(model_path)

# Chúng ta cần mô hình mà không có lớp cuối cùng (lớn softmax) để lấy vector đặc trưng
# Lớn "flatten" là một lựa chọn tốt
feature_extractor_model = Model(
    inputs=base_model.input, outputs=base_model.get_layer("flatten").output
)

# 2. Chuẩn bị danh sách ảnh và đường dẫn
image_paths = []
for root, dirs, files in os.walk("flowers"):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

# 3. Trích xuất đặc trưng
features = []
image_names = []

print("Bắt đầu trích xuất đặc trưng...")
for image_path in tqdm(image_paths):
    try:
        # Tải và tiền xử lý ảnh
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Rescale

        # Dự đoán để lấy vector đặc trưng
        feature_vector = feature_extractor_model.predict(image, verbose=0)

        # Làm phẳng vector và thêm vào danh sách
        features.append(feature_vector.flatten())
        image_names.append(image_path)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")

# 4. Lưu lại
features = np.array(features)
np.save("image_features.npy", features)
np.save("image_names.npy", image_names)

print(f"Đã trích xuất và lưu {len(features)} vector đặc trưng.")

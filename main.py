# main.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import uuid

# 1. Khởi tạo FastAPI
app = FastAPI()


# 2. Tải các thành phần cần thiết khi khởi động
@app.on_event("startup")
def load_models_and_data():
    global feature_extractor_model, image_features, image_names

    # Tải mô hình
    base_model = tf.keras.models.load_model("flower_classifier_final.keras")
    feature_extractor_model = Model(
        inputs=base_model.input, outputs=base_model.get_layer("flatten").output
    )

    # Tải vector database
    image_features = np.load("image_features.npy")
    image_names = np.load("image_names.npy")
    print("Mô hình và dữ liệu đã được tải sẵn sàng!")


# 3. Định nghĩa endpoint tìm kiếm
@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    try:
        # Tạo một tên file tạm thời duy nhất
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        contents = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(contents)

        # Tiền xử lý ảnh upload
        image = load_img(temp_filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # Trích xuất đặc trưng của ảnh upload
        query_feature = feature_extractor_model.predict(image, verbose=0)

        # Tính toán độ tương đồng cosine
        similarities = cosine_similarity(query_feature, image_features).flatten()

        # Lấy ra 5 chỉ số của ảnh có độ tương đồng cao nhất
        top_5_indices = similarities.argsort()[-5:][::-1]

        # Lấy đường dẫn ảnh và độ tương đồng
        results = []
        for index in top_5_indices:
            results.append(
                {
                    "image_path": image_names[index],
                    "similarity": float(similarities[index]),
                }
            )

        # Xóa file tạm
        os.remove(temp_filename)

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 4. Chạy server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

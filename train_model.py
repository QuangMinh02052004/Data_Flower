# train_model_optimized.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

# ==========================================================
# 1. CẤU HÌNH VÀ TỐI ƯU HÓA
# ==========================================================

# Kích hoạt Mixed Precision
# Tự động sử dụng float16 thay vì float32 khi có thể, tăng tốc đáng kể trên GPU M2
try:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled!")
except:
    print("Could not enable mixed precision.")

# Cấu hình
INIT_LR = 1e-4
EPOCHS_PHASE_1 = 10  # Giai đoạn 1: chỉ huấn luyện head
EPOCHS_PHASE_2 = 10  # Giai đoạn 2: fine-tune một phần base model
BS = 32  # Batch Size. Với 16GB RAM, bạn có thể thử tăng lên 64 nếu không bị lỗi
IMAGE_SIZE = (224, 224)
DATA_DIR = "flowers"
MODEL_PATH_PHASE_1 = "flower_classifier_phase_1.model"
FINAL_MODEL_PATH = "flower_classifier_final.model"

# ==========================================================
# 2. CHUẨN BỊ DỮ LIỆU VỚI tf.data (Tối ưu hơn ImageDataGenerator)
# ==========================================================

# Lấy danh sách các file ảnh và nhãn
image_paths = []
labels = []
class_names = sorted(os.listdir(DATA_DIR))
for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_name)

# Chuyển nhãn thành số
label_to_index = {name: index for index, name in enumerate(class_names)}
numeric_labels = [label_to_index[label] for label in labels]

# Tạo tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, numeric_labels))


# Hàm để đọc và tiền xử lý ảnh
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


# Áp dụng hàm tiền xử lý, xáo trộn, chia batch và prefetch
# AUTOTUNE sẽ tự động tối ưu số lượng process song song
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(image_paths))
dataset = dataset.batch(BS)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# Chia dataset thành train và validation
train_size = int(0.8 * len(image_paths))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

print(f"Số lớp: {len(class_names)}")
print(f"Số ảnh train: {train_size}")
print(f"Số ảnh validation: {len(image_paths) - train_size}")
np.save("class_indices.npy", label_to_index)


# ==========================================================
# 3. XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH (2 GIAI ĐOẠN)
# ==========================================================

# --- Giai đoạn 1: Huấn luyện chỉ phần "head" ---

print("\n--- Bắt đầu Giai đoạn 1: Huấn luyện head model ---")

baseModel = MobileNetV2(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)
baseModel.trainable = False  # Đóng băng toàn bộ base model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
# Lớp cuối cùng cần có dtype='float32' để ổn định với mixed precision
predictions = Dense(len(class_names), activation="softmax", dtype="float32")(headModel)

model = Model(inputs=baseModel.input, outputs=predictions)

optimizer = Adam(learning_rate=INIT_LR)
# Sử dụng loss function chuyên dụng cho mixed precision
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# Callbacks
# ReduceLROnPlateau: tự động giảm learning rate khi validation loss không cải thiện
lr_reducer = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6, verbose=1
)
# ModelCheckpoint: tự động lưu model tốt nhất
checkpoint = ModelCheckpoint(
    MODEL_PATH_PHASE_1,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

H1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE_1,
    callbacks=[lr_reducer, checkpoint],
)

# --- Giai đoạn 2: Fine-tuning một phần base model ---

print("\n--- Bắt đầu Giai đoạn 2: Fine-tuning base model ---")

# Tải lại model tốt nhất từ giai đoạn 1
model = tf.keras.models.load_model(MODEL_PATH_PHASE_1)

# Mở băng (unfreeze) các lớp cuối cùng của base model
# Chúng ta sẽ fine-tune từ lớp thứ 120 trở đi (MobileNetV2 có khoảng 154 lớp)
baseModel = model.layers[0]
baseModel.trainable = True
for layer in baseModel.layers[:120]:
    layer.trainable = False

# Compile lại với learning rate thấp hơn rất nhiều
model.compile(
    optimizer=Adam(learning_rate=INIT_LR / 10), loss=loss_fn, metrics=["accuracy"]
)

# Callback cho giai đoạn 2
final_checkpoint = ModelCheckpoint(
    FINAL_MODEL_PATH, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)

H2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE_2,
    callbacks=[final_checkpoint],
)

print("\nHuấn luyện hoàn tất! Mô hình cuối cùng đã được lưu.")

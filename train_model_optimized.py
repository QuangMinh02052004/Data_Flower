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
try:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled!")
except:
    print("Could not enable mixed precision.")

# Cấu hình
INIT_LR = 1e-4
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 20  # Đã tăng lên 20 theo gợi ý
BS = 32
IMAGE_SIZE = (224, 224)
DATA_DIR = "flowers"
MODEL_PATH_PHASE_1 = "flower_classifier_phase_1.keras"
FINAL_MODEL_PATH = "flower_classifier_final.keras"

# ==========================================================
# 2. CHUẨN BỊ DỮ LIỆU VỚI tf.data
# ==========================================================

# ... (Phần chuẩn bị dữ liệu giữ nguyên) ...
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

label_to_index = {name: index for index, name in enumerate(class_names)}
numeric_labels = [label_to_index[label] for label in labels]

dataset = tf.data.Dataset.from_tensor_slices((image_paths, numeric_labels))


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(image_paths))
dataset = dataset.batch(BS)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

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
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
baseModel.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
predictions = Dense(len(class_names), activation="softmax", dtype="float32")(headModel)

model = Model(inputs=baseModel.input, outputs=predictions)

optimizer = Adam(learning_rate=INIT_LR)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

lr_reducer = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6, verbose=1
)
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

model = tf.keras.models.load_model(MODEL_PATH_PHASE_1)

# ĐÃ SỬA LẦN CUỐI: Cách tiếp cận mạnh mẽ cho mô hình đã được làm phẳng
model.trainable = True
for layer in model.layers[:120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=INIT_LR / 10), loss=loss_fn, metrics=["accuracy"]
)

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

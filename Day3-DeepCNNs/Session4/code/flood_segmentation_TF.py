import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================================
# 1. Configuration
# ======================================
DATA_ROOT = "/path/to/datasets/Flood Area Segmentation"   # <-- set this (contains Image/ and Mask/)
IMAGE_DIR = os.path.join(DATA_ROOT, "Image")
MASK_DIR = os.path.join(DATA_ROOT, "Mask")
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"

print(f"Running on {DEVICE}")

# ======================================
# 2. Custom Dice loss & metric
# ======================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# ======================================
# 3. Data Preprocessing & Augmentation
# ======================================
data_gen_args = dict(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Generator that yields (image, mask) pairs
def image_mask_generator(image_dir, mask_dir, batch_size):
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        seed=1
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode="grayscale",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        seed=1
    )
    return zip(image_generator, mask_generator)

# ======================================
# 4. Model Architectures
# ======================================

# ---- SimpleSegNet ----
def build_simple_segnet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Decoder
    x = layers.Conv2DTranspose(16, 2, strides=2, activation="relu")(x)
    x = layers.Conv2DTranspose(1, 2, strides=2, activation="sigmoid")(x)

    model = models.Model(inputs, x, name="SimpleSegNet")
    return model

# ---- Transfer Learning (VGG16) ----
def build_vgg16_transfer(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False  # freeze conv layers

    x = base_model.output
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(128, 2, strides=2, activation="relu")(x)
    x = layers.Conv2DTranspose(64, 2, strides=2, activation="relu")(x)
    x = layers.Conv2DTranspose(32, 2, strides=2, activation="relu")(x)
    x = layers.Conv2DTranspose(16, 2, strides=2, activation="relu")(x)
    x = layers.Conv2DTranspose(1, 2, strides=2, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=x, name="VGG16_TransferSegNet")
    return model

# ======================================
# 5. Compile and Train Model
# ======================================
# Choose model: SimpleSegNet or VGG16_Transfer
model_choice = "SimpleSegNet"  # change to "VGG16" for transfer learning

if model_choice == "SimpleSegNet":
    model = build_simple_segnet(input_shape=(*IMG_SIZE, 3))
else:
    model = build_vgg16_transfer(input_shape=(*IMG_SIZE, 3))

model.compile(
    optimizer=optimizers.Adam(LR),
    loss=dice_loss,
    metrics=[dice_coefficient, "accuracy"]
)

model.summary()

# Dummy split of data — in practice, use tf.data.Dataset or custom loader
train_gen = image_mask_generator(IMAGE_DIR, MASK_DIR, BATCH_SIZE)
steps_per_epoch = len(os.listdir(os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0]))) // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS
)

# ======================================
# 6. Visualize Training Performance
# ======================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Dice Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history.history["dice_coefficient"], label="Dice Coeff.")
plt.legend()
plt.title("Dice Coefficient Curve")
plt.show()

# ======================================
# 7. Save best model
# ======================================
model.save("best_model.h5")
print("Model saved as best_model.h5")

import os
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Đường dẫn đến tập dữ liệu
data_dir = 'data/'

# Kích thước đầu vào cho từng mô hình
input_sizes = {
    'resnet50': (224, 224),
    'mobilenetv3': (224, 224),
    'inceptionv3': (299, 299),
    'efficientnetb0': (300, 300)
}

# Tạo data generator với các phép tăng cường dữ liệu
def create_data_generator(model_name, batch_size=32):
    input_size = input_sizes.get(model_name)
    
    # Chọn rescale tùy theo mô hình
    if model_name in ['resnet50', 'mobilenetv3', 'inceptionv3']:
        rescale_value = 1./127.5 - 1  # Rescale vào khoảng [-1, 1]
    else:
        rescale_value = 1./255  # Rescale vào khoảng [0, 1]

    datagen = ImageDataGenerator(
        rescale=rescale_value,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2  # 80% train, 20% validation
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator
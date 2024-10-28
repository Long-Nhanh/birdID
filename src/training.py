import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tensorflow.keras.applications import ResNet50, MobileNetV3Large, InceptionV3, EfficientNetB0  # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from preprocessing import create_data_generator  # type: ignore

# Danh sách các mô hình và kích thước đầu vào tương ứng
MODELS = {
    'resnet50': (ResNet50, (224, 224, 3)),
    'mobilenetv3': (MobileNetV3Large, (224, 224, 3)),
    'inceptionv3': (InceptionV3, (299, 299, 3)),
    'efficientnetb0': (EfficientNetB0, (300, 300, 3))
}

# Xây dựng mô hình dựa trên base model
def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Thêm Dropout để giảm overfitting
    x = Dense(107, activation='softmax', kernel_regularizer=l2(0.01))(x)  # L2 Regularization
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Huấn luyện một mô hình
def train_model(model_name, input_shape):
    base_model_class, input_size = MODELS[model_name]
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_size)
    
    model = build_model(base_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Tạo generator
    train_generator, validation_generator = create_data_generator(model_name)

    # Callback: Early stopping và ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Huấn luyện mô hình
    history = model.fit(train_generator, epochs=30, validation_data=validation_generator,
                        callbacks=[early_stopping, reduce_lr])

    # Lưu mô hình
    model.save(f'models/{model_name}_model.h5')

    return history

# Hàm vẽ quá trình huấn luyện và đánh giá
def plot_history(history, model_name):
    # Vẽ accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Vẽ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'logs/{model_name}_training.png')
    # plt.show()

# Huấn luyện tất cả các mô hình
def train_all_models():
    for model_name, (base_model_class, input_size) in MODELS.items():
        print(f"Training model: {model_name}")
        history = train_model(model_name, input_size)
        plot_history(history, model_name)

if __name__ == '__main__':
    train_all_models()
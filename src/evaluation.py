import tensorflow as tf # type: ignore
from src.preprocessing import create_data_generator

def evaluate_model(model_path, model_name):
    model = tf.keras.models.load_model(model_path)
    _, validation_generator = create_data_generator(model_name)

    loss, accuracy = model.evaluate(validation_generator)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    model_name = 'resnet50'  # Hoặc 'mobilenetv3', 'inceptionv3', 'efficientnetb0'
    model_path = f'models/{model_name}_model.h5'
    evaluate_model(model_path, model_name)

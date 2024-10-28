import os
import numpy as np  # type: ignore
from flask import Flask, render_template, request, redirect, url_for, send_from_directory  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore

app = Flask(__name__)

# Đường dẫn lưu trữ file upload
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load các mô hình
models = {
    'resnet50': load_model('models/resnet50_model.h5'),
    'mobilenetv3': load_model('models/mobilenetv3_model.h5'),
    # 'inceptionv3': load_model('models/inceptionv3_model.h5'),
    # 'efficientnetb0': load_model('models/efficientnetb0_model.h5')
}

# Tên các loài chim trong dataset
bird_species = ['Ashy Drongo', 'Ashy Tailorbird', 'Asian Brown', 'Asian Dowitcher', 'Asian Green Bee']

# Hàm tiền xử lý ảnh đầu vào theo từng mô hình
def preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(224, 224))  # Default size for MobileNetV3 and ResNet50
    if model_name == 'inceptionv3':
        img = image.load_img(img_path, target_size=(299, 299))
    elif model_name == 'efficientnetb0':
        img = image.load_img(img_path, target_size=(300, 300))
    
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if model_name in ['resnet50', 'mobilenetv3', 'inceptionv3']:
        img = img / 127.5 - 1  # Normalize for these models

    return img

# Route trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý upload ảnh và dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        model_name = request.form.get('model')
        model = models[model_name]

        # Tiền xử lý ảnh
        img = preprocess_image(filepath, model_name)

        # Dự đoán
        preds = model.predict(img)
        result = {bird_species[i]: float(preds[0][i]) for i in range(len(bird_species))}

        return render_template('index.html', result=result, filename=filename, model_name=model_name)

# Route để hiển thị ảnh đã upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

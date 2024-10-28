import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
from PIL import Image

app = Flask(__name__)

# Đường dẫn tới các mô hình
MODEL_PATHS = {
    'mobilenet': 'models/mobilenet_model.h5',
    'resnet': 'models/resnet_model.h5',
    'efficientnet': 'models/efficientnet_model.h5'
}

# Các nhãn loài chim
species_labels = ['Ashy-Drongo', 'Ashy-Tailorbird', 'Asian-Brown', 'Asian-Dowitcher', 'Asian-Green-Bee-eater']

# Hàm dự đoán
def model_predict(img_path, model):
    img = Image.open(img_path).resize((224, 224))
    img = np.array(img) / 255.0  # Chuẩn hóa ảnh
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]  # Lấy kết quả dự đoán

    return preds


# Giả sử bạn có một dictionary chứa thông tin về loài chim
bird_info = {
    'Asian-Dowitcher': {
        'name': 'Asian-Dowitcher',
        'scientific_name': 'Passeridae',
        'conservation_status': 'Least Concern',
        'habitat': 'Urban areas, woodlands, grasslands',
        'diet': 'Seeds, insects',
        'size': '16 cm',
        'image': 'sparrow.jpg',
        'description': 'Sparrows are small, plump, brown and grey birds with short tails and stubby, powerful beaks.'
    },
    'eagle': {
        'name': 'Eagle',
        'scientific_name': 'Accipitridae',
        'conservation_status': 'Least Concern',
        'habitat': 'Mountains, forests, open plains',
        'diet': 'Carnivorous',
        'size': '70-100 cm',
        'image': 'eagle.jpg',
        'description': 'Eagles are large birds of prey with strong, hooked beaks and powerful talons.'
    }
    # Thêm các loài chim khác ở đây
}

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Kiểm tra nếu người dùng có tải file lên
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Lựa chọn mô hình
        model_choice = request.form.get('model')  # Lấy mô hình được chọn
        if model_choice not in MODEL_PATHS:
            return "Invalid model choice."

        # Tải mô hình tương ứng
        model = load_model(MODEL_PATHS[model_choice])

        if file:
            # Lưu file ảnh
            file_path = os.path.join('static/img', file.filename)
            file.save(file_path)

            # Dự đoán loài chim
            preds = model_predict(file_path, model)

            # Lấy loài có xác suất cao nhất
            top_prediction = species_labels[np.argmax(preds)]

            # Chuẩn bị dữ liệu hiển thị xác suất cho từng loài
            prediction_confidences = {species_labels[i]: float(preds[i]) for i in range(len(species_labels))}

            # Trả về kết quả
            return render_template(
                'index.html',
                prediction=top_prediction,
                prediction_confidences=prediction_confidences,
                img_path=file_path
            )
        
@app.route('/upload')
def index():
    return render_template('upload.html')

@app.route('/bird/<bird_name>')
def bird_detail(bird_name):
    bird = bird_info.get(bird_name)
    if bird:
        return render_template('bird_detail.html', 
                               bird_name=bird['name'], 
                               scientific_name=bird['scientific_name'], 
                               conservation_status=bird['conservation_status'], 
                               habitat=bird['habitat'], 
                               diet=bird['diet'], 
                               size=bird['size'], 
                               bird_image=url_for('static', filename='images/' + bird['image']),
                               bird_description=bird['description'])
    else:
        return "Bird not found", 404

if __name__ == '__main__':
    app.run(debug=True)

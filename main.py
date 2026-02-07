from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# ---------------- INIT APP ----------------
app = Flask(__name__)

# ---------------- LOAD MODEL (IMAGE KE HISAB SE) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.h5")   # 👈 SAME FOLDER
model = load_model(model_path, compile=False)
print("✅ model.h5 loaded successfully")

# ---------------- CLASS LABELS ----------------
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# ---------------- UPLOAD FOLDER ----------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- PREDICTION FUNCTION ----------------
def predict_tumor(image_path):
    IMAGE_SIZE = 128

    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = float(np.max(predictions))

    label = class_labels[predicted_class_index]

    if label == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor: {label}", confidence_score

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != "":
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence*100:.2f}%",
                file_path=f'/uploads/{file.filename}'
            )

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)

import os
import cv2
from flask import Flask, request, render_template, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from PIL import Image
import torch

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Konfiguracja dla folderu uploadów
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inicjalizacja modelu YOLO (użycie modelu YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Możesz użyć 'yolov5m' lub innego modelu

# Funkcja do sprawdzenia, czy plik ma dozwolony typ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Detekcja obiektów
        result, analysis = detect_objects(filepath)

        return render_template('index.html', result=result, analysis=analysis, filename=filename)


def detect_objects(image_path):
    # Wczytaj obraz
    img = Image.open(image_path)

    # Przetwórz obraz modelem
    results = model(img)

    # Pobierz dane detekcji jako DataFrame
    pred_df = results.pandas().xywh[0]

    # Debugowanie wyników
    print("Prediction DataFrame:")
    print(pred_df)

    # Sprawdź, czy jakiekolwiek obiekty zostały wykryte
    if pred_df.empty:
        print("No objects detected.")
        return "No objects detected.", "Brak wykrytych obiektów."

    # Filtruj dla klas 'apple' (klasa 47) i 'orange' (klasa 49)
    filtered = pred_df[pred_df['class'].isin([47, 49])]

    if filtered.empty:
        print("No apples or oranges detected.")
        return "No apples or oranges detected.", "Brak wykrytych jabłek lub pomarańczy."

    # Twórz analizę
    analysis = "\n".join(
        f"Obiekt: {row['name']}, Pewność: {row['confidence']:.2f}, Pozycja: x={row['xcenter']}, y={row['ycenter']}, width={row['width']}, height={row['height']}"
        for _, row in filtered.iterrows()
    )

    # Twórz podsumowanie
    detected_objects = filtered['name'].value_counts().to_dict()
    result = "Wykryto: " + ", ".join(f"{count}x {name}" for name, count in detected_objects.items())

    print("Analysis:")
    print(analysis)

    return result, analysis

print("Classes in model:")
print(model.names)

if __name__ == '__main__':
    app.run(debug=True)

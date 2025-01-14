import cv2
import torch
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Ustawienia ścieżek dla przesłanych i wynikowych obrazów
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Załaduj model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def resize_image(image, target_size=(640, 640)):
    """
    Skaluje obraz do zadanego rozmiaru.

    Args:
    - image: Obraz w formacie numpy array.
    - target_size: Krotka określająca rozmiar docelowy (szerokość, wysokość).

    Returns:
    - Zmieniony obraz o zadanym rozmiarze.
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def draw_boxes_with_labels(image, detections):
    """
    Rysuje prostokąty wokół obiektów i numeruje je.

    Args:
    - image: Obraz w formacie numpy array.
    - detections: Wyniki detekcji z modelu YOLOv5.

    Returns:
    - Annotowany obraz.
    - Słownik liczący wykryte obiekty oraz ich szczegóły.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (0, 255, 0)  # Zielony kolor prostokąta i tekstu
    count = {"apple": 0, "orange": 0}  # Licznik obiektów
    details = []  # Szczegóły dotyczące obiektów

    for det in detections:
        label = det['name']
        confidence = det['confidence']
        if label in count:
            count[label] += 1
            x_min, y_min, x_max, y_max = map(int, det['box'])
            # Rysowanie prostokąta
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            # Dodanie etykiety
            label_text = f"{label} #{count[label]} ({confidence:.2f})"
            cv2.putText(image, label_text, (x_min, y_min - 10), font, font_scale, color, font_thickness)
            details.append({"label": label, "confidence": confidence, "id": count[label]})

    return image, count, details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Wczytaj obraz
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img_np = np.array(img)

        # Skaluj obraz do zalecanego rozmiaru
        scaled_img = resize_image(img_np)

        # Wykonaj detekcję obiektów
        results = model(scaled_img)

        # Przetwórz wyniki detekcji
        detections = []
        for *box, conf, cls in results.xyxy[0].tolist():
            label = model.names[int(cls)]
            if label in ['apple', 'orange']:
                detections.append({'name': label, 'box': box, 'confidence': conf})

        # Rysuj prostokąty i etykiety
        annotated_img, count, details = draw_boxes_with_labels(scaled_img, detections)

        # Zapisz wynikowy obraz
        image_name = file.filename.rsplit('.', 1)[0]
        result_image_path = os.path.join(UPLOAD_FOLDER, f"{image_name}_result.jpg")
        cv2.imwrite(result_image_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

        # Przekaż dane do szablonu
        return render_template(
            'result.html',
            result_image=result_image_path,
            apple_count=count['apple'],
            orange_count=count['orange'],
            total_count=count['apple'] + count['orange'],
            details=details
        )

if __name__ == '__main__':
    app.run(debug=True)

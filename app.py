from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np

app = Flask(__name__)

# Set folder upload dan folder hasil video
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Inisialisasi YOLO model dan konfigurasi zona
model = YOLO(r"best.pt")
classnames = ['background', 'car', 'truck', 'bus']

zone1 = np.array([[133, 201], [271, 206], [265, 329], [4, 286], [133, 201]], np.int32)
zone2 = np.array([[281, 204], [452, 188], [639, 269], [286, 298], [281, 204]], np.int32)

zone1_line = np.array([zone1[0], zone1[1]]).reshape(-1)
zone2_line = np.array([zone2[0], zone2[1]]).reshape(-1)

tracker = Sort()

# Fungsi untuk memeriksa tipe file yang diunggah
def allowed_file(filename):
    """Cek apakah file memiliki ekstensi yang diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    """Render halaman upload."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Proses unggahan video."""
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Proses video
        processed_video_path, zone_a_count, zone_b_count = process_video(file_path)
        
        return render_template(
            'hasil.html',
            filename=os.path.basename(processed_video_path),
            zone_a_count=zone_a_count,
            zone_b_count=zone_b_count
        )

    return "Invalid file type", 400

@app.route('/download/<filename>')
def download_file(filename):
    """Unduh video yang telah diproses."""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

def process_video(file_path):
    """Proses video menggunakan YOLO dan SORT."""
    zoneAcounter = set()
    zoneBcounter = set()
    
    # Buka video
    cap = cv2.VideoCapture(file_path)
    processed_filename = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_filename, fourcc, 30.0, (640, 544))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 544))

        # Jalankan YOLO pada frame
        results = model(frame, verbose=False)
        current_detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_detect = classnames[class_id]

                if class_detect in ['car', 'truck', 'bus'] and confidence > 0.7:
                    current_detections.append([x1, y1, x2, y2, confidence, class_id])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Gambarkan zona pada frame
        cv2.polylines(frame, [zone1], isClosed=False, color=(0, 0, 255), thickness=1)
        cv2.polylines(frame, [zone2], isClosed=False, color=(0, 255, 255), thickness=1)

        # Update tracker jika ada deteksi
        if current_detections:
            track_results = tracker.update(np.array(current_detections))
        else:
            track_results = np.empty((0, 5))

        # Hitung kendaraan di setiap zona
        for track in track_results:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2 - 40

            if zone1_line[0] < cx < zone1_line[2] and zone1_line[1] - 20 < cy < zone1_line[1] + 20:
                zoneAcounter.add(track_id)

            if zone2_line[0] < cx < zone2_line[2] and zone2_line[1] - 30 < cy < zone2_line[1] + 30:
                zoneBcounter.add(track_id)

        # Tambahkan teks pada frame
        cv2.putText(frame, f"Zone A: {len(zoneAcounter)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Zone B: {len(zoneBcounter)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(frame)

    cap.release()
    out.release()
    
    return processed_filename, len(zoneAcounter), len(zoneBcounter)

if __name__ == '__main__':
    app.run(debug=True)
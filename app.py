import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from detector import detect_image
from video_utils import detect_video
from database import init_db, save_scan, get_all_scans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_IMAGES = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_VIDEOS = {'mp4', 'avi', 'mov'}
ALLOWED_ALL    = ALLOWED_IMAGES | ALLOWED_VIDEOS

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
init_db()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_ALL

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return 'video' if ext in ALLOWED_VIDEOS else 'image'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected.')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')
    if not allowed_file(file.filename):
        return render_template('index.html',
               error='Unsupported file type. Upload JPG, PNG or MP4.')
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    file_type = get_file_type(filename)
    if file_type == 'video':
        detection = detect_video(save_path)
    else:
        detection = detect_image(save_path)
    save_scan(filename, file_type, detection['result'],
              detection['confidence'])
    os.remove(save_path)
    return render_template('result.html',
        filename=filename,
        file_type=file_type,
        result=detection['result'],
        percent=detection['percent'],
        confidence=detection['confidence'],
        eye_analysis=detection.get('eye_analysis', 'N/A'),
        skin_analysis=detection.get('skin_analysis', 'N/A'),
        face_geometry=detection.get('face_geometry', 'N/A'),
        frequency=detection.get('frequency', 'N/A'),
        efficientnet_score=detection.get('efficientnet_score', 'N/A'),
        resnet_score=detection.get('resnet_score', 'N/A'),
        svm_score=detection.get('svm_score', 'N/A')
    )

@app.route('/history')
def history():
    scans = get_all_scans()
    return render_template('history.html', scans=scans)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
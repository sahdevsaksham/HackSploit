import base64
import io
import os
import random
import sys
import time
from flask import Flask, jsonify, render_template, request, send_from_directory
import cv2 as cv
import numpy as np
from PIL import Image
import torch

# Ensure local packages can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from facenet.models.mtcnn import MTCNN
from liveness_detection.blink_detection import BlinkDetector
from liveness_detection.emotion_prediction import EmotionPredictor
from liveness_detection.face_orientation import FaceOrientationDetector
from utils.distance import Euclidean_Distance, findThreshold
from utils.functions import extract_face, face_transform, get_image
from verification_models import VGGFace2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Global device and models state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing eKYC models on device: {DEVICE}...")

# Load models
mtcnn_detector = MTCNN(device=DEVICE)
vggface2_model = VGGFace2.load_model(device=DEVICE)
vggface2_model.eval()

blink_detector = BlinkDetector()
orientation_detector = FaceOrientationDetector()
emotion_predictor = EmotionPredictor(device=DEVICE)

print("[✓] All eKYC AI Models loaded successfully!")

# In-memory session state for single-user/local testing
session_data = {
    "id_image": None,
    "id_cropped_face": None,
    "id_embedding": None,
    "last_live_face": None,
    "last_distance": None,
    "verified": False,
    "challenges": [],
    "current_challenge_index": 0,
    "blink_counter": 0,
    "blink_total": 0,
    "completed_challenges": []
}


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 data string to an RGB NumPy array."""
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image from base64 string")
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


def encode_rgb_to_base64(img_rgb: np.ndarray, quality=90) -> str:
    """Encode an RGB NumPy array to a base64 JPEG data URL."""
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    _, buffer = cv.imencode('.jpg', img_bgr, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    return jsonify({
        "status": "ready",
        "device": str(DEVICE),
        "models": {
            "face_detector": "MTCNN",
            "face_verification": "InceptionResnetV1 (VGGFace2)",
            "blink_detector": "Dlib 68-Landmark EAR",
            "orientation_detector": "MTCNN 5-Landmark Angle Vector",
            "emotion_predictor": "VGG-Face Emotion Classifier"
        },
        "session": {
            "has_id": session_data["id_embedding"] is not None,
            "verified": session_data["verified"]
        }
    })


@app.route('/api/sample/<filename>')
def serve_sample(filename):
    samples_dir = os.path.join(BASE_DIR, 'static', 'samples')
    return send_from_directory(samples_dir, filename)


@app.route('/api/load_sample', methods=['POST'])
def load_sample():
    data = request.get_json() or {}
    sample_key = data.get('sample', 'sample1')

    if sample_key == 'sample1':
        sample_path = os.path.join(BASE_DIR, 'static', 'samples', 'sample_id_specimen.jpg')
    else:
        sample_path = os.path.join(BASE_DIR, 'static', 'samples', 'sample_id_specimen2.jpg')

    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": f"Sample image {sample_path} not found"}), 404

    img_rgb = get_image(sample_path)
    return process_id_image(img_rgb)


@app.route('/api/upload_id', methods=['POST'])
def upload_id():
    img_rgb = None
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        in_memory = io.BytesIO()
        file.save(in_memory)
        in_memory.seek(0)
        file_bytes = np.asarray(bytearray(in_memory.read()), dtype=np.uint8)
        img_bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Invalid image file"}), 400
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    else:
        data = request.get_json() or {}
        image_b64 = data.get('image')
        if not image_b64:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        try:
            img_rgb = decode_base64_image(image_b64)
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to decode image: {str(e)}"}), 400

    return process_id_image(img_rgb)


def process_id_image(img_rgb: np.ndarray):
    """Detects face in the ID card image and extracts its embedding."""
    try:
        face, box, landmarks = extract_face(img_rgb, mtcnn_detector, padding=1)
        if box is None or face is None or face.size == 0:
            return jsonify({
                "success": False,
                "error": "No face detected in the document. Please upload a clear photo with a visible face."
            }), 400

        # Calculate embedding
        with torch.no_grad():
            face_tensor = face_transform(face, model_name="VGG-Face2", device=DEVICE)
            embedding = vggface2_model(face_tensor)

        # Store in session
        session_data["id_image"] = img_rgb
        session_data["id_cropped_face"] = face
        session_data["id_embedding"] = embedding
        session_data["verified"] = False

        cropped_b64 = encode_rgb_to_base64(face)
        full_b64 = encode_rgb_to_base64(img_rgb)

        return jsonify({
            "success": True,
            "face_detected": True,
            "box": [int(b) for b in box],
            "cropped_face": cropped_b64,
            "full_image": full_b64,
            "message": "ID face successfully extracted and vectorized!"
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Face extraction error: {str(e)}"}), 500


@app.route('/api/verify_face', methods=['POST'])
def verify_face():
    """Compares live camera frame with uploaded ID card face."""
    if session_data["id_embedding"] is None:
        return jsonify({
            "success": False,
            "error": "No ID card uploaded yet. Please complete Step 1 first."
        }), 400

    data = request.get_json() or {}
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({"success": False, "error": "No frame provided"}), 400

    try:
        live_rgb = decode_base64_image(image_b64)
        live_face, box, landmarks = extract_face(live_rgb, mtcnn_detector, padding=1)

        if box is None or live_face is None or live_face.size == 0:
            return jsonify({
                "success": True,
                "face_detected": False,
                "verified": False,
                "message": "No face found in camera view. Please center your face."
            })

        # Calculate embedding
        with torch.no_grad():
            live_tensor = face_transform(live_face, model_name="VGG-Face2", device=DEVICE)
            live_embedding = vggface2_model(live_tensor)

            # Euclidean distance
            distance = float(Euclidean_Distance(session_data["id_embedding"], live_embedding))

        # Threshold from distance.py (0.70 for VGG-Face2 euclidean)
        threshold = findThreshold(model_name="VGG-Face2", distance_metric="euclidean")
        verified = distance < threshold

        # Update session
        session_data["last_live_face"] = live_face
        session_data["last_distance"] = distance
        if verified:
            session_data["verified"] = True

        live_face_b64 = encode_rgb_to_base64(live_face)

        return jsonify({
            "success": True,
            "face_detected": True,
            "verified": verified,
            "distance": round(distance, 4),
            "threshold": threshold,
            "box": [int(b) for b in box],
            "live_face": live_face_b64,
            "message": "Face match successful!" if verified else "Face does not match ID document"
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Verification error: {str(e)}"}), 500


@app.route('/api/challenge/new', methods=['GET', 'POST'])
def new_challenges():
    """Generates 3 randomized liveness challenges."""
    challenge_pool = [
        {
            "type": "emotion",
            "target": "smile",
            "title": "Please Smile",
            "instruction": "Show a clear, genuine smile towards the camera",
            "icon": "😃"
        },
        {
            "type": "emotion",
            "target": "surprise",
            "title": "Look Surprised",
            "instruction": "Open your mouth or widen your eyes in a surprised expression",
            "icon": "😲"
        },
        {
            "type": "orientation",
            "target": "left",
            "title": "Turn Face Left",
            "instruction": "Gently turn your head towards your LEFT",
            "icon": "👈"
        },
        {
            "type": "orientation",
            "target": "right",
            "title": "Turn Face Right",
            "instruction": "Gently turn your head towards your RIGHT",
            "icon": "👉"
        },
        {
            "type": "blink",
            "target": random.randint(2, 3),
            "title": "Blink Your Eyes",
            "instruction": "Blink your eyes 2 times slowly in front of the camera",
            "icon": "😉"
        }
    ]

    selected = random.sample(challenge_pool, 3)
    # Ensure targets are properly serialized
    session_data["challenges"] = selected
    session_data["current_challenge_index"] = 0
    session_data["completed_challenges"] = []
    session_data["blink_total"] = 0
    session_data["blink_counter"] = 0
    blink_detector.total = 0
    blink_detector.counter = 0

    return jsonify({
        "success": True,
        "challenges": selected,
        "current_index": 0
    })


@app.route('/api/challenge/evaluate', methods=['POST'])
def evaluate_challenge():
    """Evaluates the live camera frame against the current challenge."""
    data = request.get_json() or {}
    image_b64 = data.get('image')
    challenge_index = data.get('index', session_data["current_challenge_index"])

    if not image_b64:
        return jsonify({"success": False, "error": "No image frame provided"}), 400

    if not session_data["challenges"] or challenge_index >= len(session_data["challenges"]):
        return jsonify({"success": False, "error": "Invalid challenge state"}), 400

    challenge = session_data["challenges"][challenge_index]
    c_type = challenge["type"]
    c_target = challenge["target"]

    try:
        frame_rgb = decode_base64_image(image_b64)
        face, box, landmarks = extract_face(frame_rgb, mtcnn_detector, padding=10)

        if box is None or face is None or face.size == 0:
            return jsonify({
                "success": True,
                "face_detected": False,
                "passed": False,
                "reading": "No Face Detected",
                "message": "Keep face visible"
            })

        passed = False
        reading = "Evaluating..."

        if c_type == "emotion":
            pred_emotion = emotion_predictor.predict(face)
            reading = pred_emotion
            passed = (pred_emotion == c_target)

        elif c_type == "orientation":
            pred_orientation = orientation_detector.detect(landmarks)
            reading = f"Turned {pred_orientation}"
            passed = (pred_orientation == c_target)

        elif c_type == "blink":
            # Use BlinkDetector
            is_blink = blink_detector.eye_blink(frame_rgb, box, thresh=c_target)
            reading = f"Blinks: {blink_detector.total} / {c_target}"
            passed = bool(is_blink)

        if passed:
            if challenge["title"] not in session_data["completed_challenges"]:
                session_data["completed_challenges"].append(challenge["title"])

        return jsonify({
            "success": True,
            "face_detected": True,
            "passed": passed,
            "reading": reading,
            "challenge_index": challenge_index,
            "blink_count": blink_detector.total if c_type == "blink" else 0,
            "blink_target": c_target if c_type == "blink" else 0
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Evaluation error: {str(e)}"}), 500


@app.route('/api/challenge/advance', methods=['POST'])
def advance_challenge():
    """Moves to next challenge or marks liveness complete."""
    session_data["current_challenge_index"] += 1
    # Reset blink detector counters for next step
    blink_detector.total = 0
    blink_detector.counter = 0

    all_done = session_data["current_challenge_index"] >= len(session_data["challenges"])
    return jsonify({
        "success": True,
        "current_index": session_data["current_challenge_index"],
        "all_done": all_done
    })


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Resets entire session state for a new verification run."""
    session_data["id_image"] = None
    session_data["id_cropped_face"] = None
    session_data["id_embedding"] = None
    session_data["last_live_face"] = None
    session_data["last_distance"] = None
    session_data["verified"] = False
    session_data["challenges"] = []
    session_data["current_challenge_index"] = 0
    session_data["completed_challenges"] = []
    blink_detector.total = 0
    blink_detector.counter = 0

    return jsonify({"success": True, "message": "Session reset successfully"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f"🚀 eKYC Server Live at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)

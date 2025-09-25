import mysql.connector
from flask import Flask, request, jsonify, send_from_directory
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import get_custom_objects
from datetime import datetime
from flask import redirect, url_for
import random
from flask_cors import CORS # ✅ เพิ่ม CORS เพื่อให้แอป Android เชื่อมต่อได้

# ✅ PyTorch Imports สำหรับโมเดลทำนายสีผิว
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

# ✅ ปิดการใช้ GPU เพื่อใช้ CPU เท่านั้น
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ ตั้งค่าการเชื่อมต่อ MySQL
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "db_miniprojectfinal"
}

# ✅ สร้าง Flask App
app = Flask(__name__)
CORS(app) # ✅ เปิดใช้งาน CORS สำหรับทุกเส้นทาง
bcrypt = Bcrypt(app)
app.config["JWT_SECRET_KEY"] = "ggygyuf6ydfyh8u5yusfuy"
jwt = JWTManager(app)

# ✅ ตั้งค่าที่เก็บไฟล์อัปโหลด
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ ลงทะเบียน Custom Layers
get_custom_objects().update({"swish": tf.keras.activations.swish})

class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

get_custom_objects().update({"FixedDropout": FixedDropout})

# ✅ โหลดโมเดล ResNet50 และใช้เป็น Feature Extractor
MODEL_PATH = "resnet50_final_model_v2_edit_1.keras"
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"swish": tf.keras.activations.swish, "FixedDropout": FixedDropout, "AdamW": AdamW}
    )
    print("✅ โหลดโมเดลสำเร็จ!")

    # ✅ ใช้ ResNet50 เป็น Feature Extractor
    resnet_base = model.get_layer("resnet50")
    resnet_base.trainable = False  # ปิดการเรียนรู้ใหม่

    feature_extractor = tf.keras.Sequential([
        resnet_base,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    print("✅ ใช้ GlobalAveragePooling2D เป็น Feature Extractor")

    # ✅ ป้องกันปัญหา BatchNormalization
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = feature_extractor.predict(dummy_input, verbose=0)

except Exception as e:
    print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    model = None
    feature_extractor = None

# ✅ โหลดฐานข้อมูลฟีเจอร์
FEATURE_DB_PATH = "feature_database.npy"
LABELS_DB_PATH = "label_database.npy"
try:
    feature_database = np.load(FEATURE_DB_PATH, allow_pickle=True)
    label_database = np.load(LABELS_DB_PATH, allow_pickle=True)
    print(f"✅ โหลดฐานข้อมูลฟีเจอร์สำเร็จ! จำนวนตัวอย่าง: {len(label_database)}")
except Exception as e:
    print(f"❌ ไม่สามารถโหลดฐานข้อมูลฟีเจอร์: {e}")
    feature_database = None
    label_database = None

# ✅ โหลดโมเดล MobileNetV2 สำหรับทำนายสีผิว (PyTorch)
skin_tone_model = None
SKIN_TONE_MODEL_PATH = "mobilenet_v2_skintonemodel.pth"
try:
    mobilenet_v2_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    num_skin_tone_classes = 4 # deep dark, fair, brown, medium

    mobilenet_v2_model.classifier[1] = torch.nn.Sequential(
        torch.nn.Identity(), # Placeholder เพื่อให้ Linear layer อยู่ที่ index 1
        torch.nn.Linear(mobilenet_v2_model.classifier[1].in_features, num_skin_tone_classes)
    )

    skin_tone_model = mobilenet_v2_model
    skin_tone_model.load_state_dict(torch.load(SKIN_TONE_MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    skin_tone_model.eval()
    print(f"✅ โหลดโมเดล MobileNetV2 สำหรับทำนายสีผิว (PyTorch) จาก {SKIN_TONE_MODEL_PATH} สำเร็จ!")

except ImportError:
    print("❌ ไม่พบ PyTorch หรือ torchvision กรุณาติดตั้ง: pip install torch torchvision")
    skin_tone_model = None
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดล MobileNetV2 สำหรับทำนายสีผิวจาก {SKIN_TONE_MODEL_PATH}: {e}")
    skin_tone_model = None

# ✅ ฟังก์ชันเชื่อมต่อ MySQL
def connect_db():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: {err}")
        return None

# ✅ ฟังก์ชันเตรียมภาพ
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img_resized = cv2.resize(img, target_size)
        img_resized = img_to_array(img_resized)
        img_resized = tf.keras.applications.resnet50.preprocess_input(img_resized)
        return np.expand_dims(img_resized, axis=0)
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดหรือปรับขนาดภาพ {image_path}: {e}")
        return None

# ✅ ฟังก์ชันดึง Feature Vector
def get_feature_vector(image_path):
    img = preprocess_image(image_path)
    if img is not None and feature_extractor is not None:
        return feature_extractor.predict(img, verbose=0)[0]
    return None

def find_top_similar_faces(test_vector, top_n=5):
    if feature_database is None or label_database is None:
        return []
    similarities = cosine_similarity(test_vector.reshape(1, -1), feature_database)[0]
    sorted_indices = similarities.argsort()[::-1]
    results = []
    seen_names = set()
    for idx in sorted_indices:
        label = label_database[idx]
        percent = round(float(similarities[idx]) * 100, 2)
        if label not in seen_names:
            results.append({"name": label, "confidence": percent, "raw_index": idx})
            seen_names.add(label)
        if len(results) == top_n:
            break
    return results

# ✅ ฟังก์ชันสำหรับแปลงคลาสสีผิวเป็นโทนความสว่างโดยรวม (Brightness Tone)
def map_class_to_brightness_tone(predicted_class):
    """
    ฟังก์ชันนี้จะแปลงคลาสสีผิวที่ทำนายได้
    ให้เป็นโทนความสว่างโดยรวมที่เข้าใจง่ายขึ้น (สว่าง, กลาง, เข้ม)
    """
    if predicted_class == "fair":
        return "โทนสว่าง"
    elif predicted_class == "medium" or predicted_class == "brown":
        return "โทนกลาง"
    elif predicted_class == "deep dark":
        return "โทนเข้ม"
    else:
        return "ไม่ระบุโทนความสว่าง"

# ✅ ฟังก์ชันสำหรับคำนวณ Undertone โดยรวม (Warm/Cool/Neutral)
def calculate_overall_undertone(all_probabilities):
    """
    ฟังก์ชันนี้จะคำนวณ Undertone โดยรวม (Warm, Cool, Neutral)
    จากเปอร์เซ็นต์ความน่าจะเป็นของแต่ละคลาสสีผิว
    """
    undertone_scores = {
        "Warm Tone": 0.0,
        "Cool Tone": 0.0,
        "Neutral Tone": 0.0
    }

    # ✅ กำหนดน้ำหนักของแต่ละคลาสสีผิวต่อ Undertone
    undertone_mapping = {
        "fair": {"Cool Tone": 0.8, "Neutral Tone": 0.2},
        "medium": {"Neutral Tone": 0.6, "Warm Tone": 0.2, "Cool Tone": 0.2},
        "brown": {"Warm Tone": 0.9, "Neutral Tone": 0.1},
        "deep dark": {"Neutral Tone": 0.7, "Cool Tone": 0.3}
    }

    for skin_class, prob_percent in all_probabilities.items():
        if skin_class in undertone_mapping:
            for undertone, weight in undertone_mapping[skin_class].items():
                undertone_scores[undertone] += (prob_percent * weight)

    if not undertone_scores or sum(undertone_scores.values()) == 0:
        return "ไม่ระบุ Undertone", {"Warm Tone": 0.0, "Cool Tone": 0.0, "Neutral Tone": 0.0}

    overall_undertone = max(undertone_scores, key=undertone_scores.get)

    total_score = sum(undertone_scores.values())
    undertone_percentages = {
        tone: round((score / total_score) * 100, 2)
        for tone, score in undertone_scores.items()
    }

    return overall_undertone, undertone_percentages

# ✅ ฟังก์ชันสำหรับการทำนายสีผิวด้วย AI Model จริงของคุณ (MobileNetV2 PyTorch)
def predict_skin_tone_from_image(image_path):
    skin_tone_categories = ["deep dark", "fair", "brown", "medium"]

    if skin_tone_model is None:
        print("⚠️ โมเดลทำนายสีผิว (MobileNetV2) ไม่ได้โหลด จะใช้การทำนายแบบสุ่ม")
        random_probs = [random.random() for _ in skin_tone_categories]
        total_sum = sum(random_probs)
        normalized_probs = [p / total_sum for p in random_probs]

        all_probabilities = {}
        for i, category in enumerate(skin_tone_categories):
            all_probabilities[category] = round(normalized_probs[i] * 100, 2)

        predicted_class = max(all_probabilities, key=all_probabilities.get)
        confidence_score = all_probabilities[predicted_class]
        return predicted_class, confidence_score, all_probabilities

    try:
        img = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = skin_tone_model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities_np = probabilities.numpy() * 100

        all_probabilities = {}
        for i, category in enumerate(skin_tone_categories):
            if i < len(probabilities_np):
                all_probabilities[category] = round(float(probabilities_np[i]), 2)
            else:
                all_probabilities[category] = 0.0

        if not all_probabilities:
            return None, None, None

        predicted_class = max(all_probabilities, key=all_probabilities.get)
        confidence_score = all_probabilities[predicted_class]

        return predicted_class, confidence_score, all_probabilities
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทำนายสีผิวด้วยโมเดล MobileNetV2: {e}")
        return None, None, None
    
# ===== helpers =====
LOOK_MAP = {
    "natural": {"keywords": ["ธรรมชาติ","natural","everyday","daily"]},
    "korean":  {"keywords": ["สายเกาหลี","เกาหลี","korean","k-beauty","k beauty"]},
    "western": {"keywords": ["สายฝอ","ฝอ","western","glam"]},
}
def canon_look(val: str):
    if not val: return None
    v = val.strip().lower()
    for k, spec in LOOK_MAP.items():
        if v in [w.lower() for w in spec["keywords"]]:
            return k
    return None

def coerce_undertone(val: str):
    if not val: return None
    v = val.strip().lower()
    if v in {"warm tone","cool tone","neutral tone"}: return v.title()
    th = {"โทนอุ่น":"Warm Tone","โทนเย็น":"Cool Tone","โทนกลาง":"Neutral Tone"}
    return th.get(val, None)

# undertone -> ความสว่างที่ “ควร” เหมาะ (ใช้ suitableSkinTone)
UNDERTONE_TO_BRIGHTNESS = {
    "Warm Tone":   ["Medium","Deep","All"],
    "Cool Tone":   ["Fair","Medium","All"],
    "Neutral Tone":["All","Fair","Medium","Deep"]
}

# ✅ API ลงทะเบียน
@app.route('/ai/register', methods=['POST'])
def register():
    data = request.get_json(silent=True)
    if not data or "username" not in data or "password" not in data:
        return jsonify({"status": "error", "message": "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"}), 400

    username = data["username"]
    password = data["password"]
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({"status": "error", "message": "ชื่อผู้ใช้นี้ถูกใช้ไปแล้ว"}), 400

        cursor.execute("INSERT INTO users (username, password, Role_ID) VALUES (%s, %s, 1)", (username, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "สมัครสมาชิกสำเร็จ!"}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ✅ API เข้าสู่ระบบ
@app.route('/ai/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"}), 400

    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            access_token = create_access_token(identity=str(user["Users_ID"]))

            return jsonify({"message": "เข้าสู่ระบบสำเร็จ!", "token": access_token}), 200
        else:
            return jsonify({"error": "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ API ทำนายใบหน้า
@app.route('/ai/predict', methods=['POST'])
@jwt_required()
def predict():
    print("🔍 Debug: Request Headers →", request.headers)
    print("🔍 Debug: Request Files →", request.files)

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    test_face_vector = get_feature_vector(file_path)
    os.remove(file_path)

    if test_face_vector is None:
        return jsonify({"error": "Failed to process image"}), 500

    top_matches = find_top_similar_faces(test_face_vector, top_n=5)

    if top_matches:
        best_match = top_matches[0]
        try:
            current_user_id = get_jwt_identity()
            conn = connect_db()
            if conn is None:
                print("❌ Error: Database connection failed for predict API")
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT ThaiCelebrities_ID FROM thaicelebrities WHERE ThaiCelebrities_name = %s", (best_match['name'],))
                celeb_row = cursor.fetchone()
                if celeb_row:
                    celeb_id = celeb_row[0]
                else:
                    celeb_id = None

                if celeb_id is not None:
                    insert_query = '''
                        INSERT INTO similarity (similarityDetail_Percent, ThaiCelebrities_ID, Users_ID, similarity_Date)
                        VALUES (%s, %s, %s, CURRENT_DATE)
                    '''
                    cursor.execute(insert_query, (
                        best_match['confidence'],
                        celeb_id,
                        int(current_user_id)
                    ))
                    conn.commit()
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"❌ Error saving best match to DB: {e}")

    return jsonify({
        "top_matches": [{"name": m["name"], "confidence": m["confidence"]} for m in top_matches],
        "message": "Top 5 most similar celebrities found."
    }), 200 if top_matches else 500

# ✅ API ทำนายสีผิว (ปรับให้ตรงสคีมา + เสถียรขึ้น)
@app.route('/ai/predict_skin_tone', methods=['POST'])
@jwt_required()
def predict_skin_tone():
    print("🔍 Debug: Skin Tone Request Headers →", request.headers)
    print("🔍 Debug: Skin Tone Request Files →", request.files)

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided for skin tone prediction"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    conn = None
    cursor = None
    try:
        file.save(file_path)

        # ทำนาย
        predicted_class, confidence, all_probabilities = predict_skin_tone_from_image(file_path)
        if predicted_class is None:
            return jsonify({"error": "Failed to predict skin tone"}), 500

        brightness_tone = map_class_to_brightness_tone(predicted_class)  # สว่าง/กลาง/เข้ม
        overall_undertone, undertone_percentages = calculate_overall_undertone(all_probabilities)  # Warm/Cool/Neutral

        # บันทึกลง DB
        current_user_id = int(get_jwt_identity())
        conn = connect_db()
        if conn is None:
            print("❌ Error: Database connection failed for skin tone API")
        else:
            cursor = conn.cursor()

            # ถ้าตารางคุณมีคอลัมน์ Undertone อยู่แล้ว ให้ใช้ INSERT แบบนี้:
            # cursor.execute(
            #   "INSERT INTO skintoneanalysis (SkinTone, Undertone, Users_ID) VALUES (%s, %s, %s)",
            #   (brightness_tone, overall_undertone, current_user_id)
            # )

            # ถ้าตารางมีแค่ SkinTone, Users_ID (ตามสคีมาปัจจุบัน) ให้เก็บ Undertone ลง SkinTone ไปก่อน:
            cursor.execute(
                "INSERT INTO skintoneanalysis (SkinTone, Users_ID) VALUES (%s, %s)",
                (overall_undertone, current_user_id)
            )

            conn.commit()
            print(f"✅ Saved SkinToneAnalysis: user={current_user_id}, tone='{overall_undertone}'")

        return jsonify({
            "overall_undertone": overall_undertone,            # Warm/Cool/Neutral
            "predicted_class": predicted_class,                # deep dark/fair/brown/medium
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "brightness_tone": brightness_tone,                # โทนสว่าง/กลาง/เข้ม
            "undertone_percentages": undertone_percentages,
            "message": "Skin tone prediction successful."
        }), 200

    except Exception as e:
        print(f"❌ predict_skin_tone error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # cleanup + close
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as _:
            pass
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ✅ API รับ Feedback จาก Android App (แก้ไขแล้ว)
@app.route('/ai/submit_feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    try:
        data = request.get_json(silent=True)
        print(f"Received JSON data for feedback: {data}")

        if not data:
            return jsonify({"status": "error", "message": "JSON ไม่ถูกต้อง"}), 400

        # -------- Parse & Validate --------
        rating = data.get('rating', None)
        feedback_text = data.get('feedback_text', "")
        cosmetic_id = data.get('cosmetic_id', None)  # optional

        # rating: allow int or float, then cast to int
        if rating is None:
            return jsonify({"status": "error", "message": "ต้องมี rating"}), 400
        if isinstance(rating, float):
            rating = int(rating)
        if not isinstance(rating, int):
            return jsonify({"status": "error", "message": "rating ต้องเป็นตัวเลขจำนวนเต็ม"}), 400
        if rating < 1 or rating > 5:
            return jsonify({"status": "error", "message": "rating ต้องอยู่ระหว่าง 1–5"}), 400

        # feedback_text: must be string; trim; cap length
        if not isinstance(feedback_text, str):
            return jsonify({"status": "error", "message": "feedback_text ต้องเป็นข้อความ"}), 400
        feedback_text = feedback_text.strip()
        if len(feedback_text) > 1000:
            feedback_text = feedback_text[:1000]

        # cosmetic_id: optional and numeric
        if cosmetic_id is not None:
            try:
                cosmetic_id = int(cosmetic_id)
                if cosmetic_id <= 0:
                    cosmetic_id = None
            except Exception:
                cosmetic_id = None

        # derive decision (optional; ใช้ภายใน response)
        decision = "skip"
        if rating >= 4:
            decision = "like"
        elif rating <= 2:
            decision = "dislike"

        # -------- DB Save --------
        user_id = int(get_jwt_identity())
        conn = cursor = None
        try:
            conn = connect_db()
            if conn is None:
                return jsonify({"status": "error", "message": "Database connection failed"}), 500
            cursor = conn.cursor()

            # ตรวจว่าตาราง feedback มีคอลัมน์ CosmeticID หรือไม่ (กันกรณีสคีมายังไม่อัปเดต)
            cursor.execute("SHOW COLUMNS FROM feedback LIKE 'CosmeticID'")
            has_cosmetic_col = cursor.fetchone() is not None

            if has_cosmetic_col and cosmetic_id:
                insert_sql = """
                    INSERT INTO feedback (Users_ID, CosmeticID, Rating, CommentText, Date)
                    VALUES (%s, %s, %s, %s, CURRENT_DATE())
                """
                args = (user_id, cosmetic_id, rating, feedback_text)
            else:
                insert_sql = """
                    INSERT INTO feedback (Users_ID, Rating, CommentText, Date)
                    VALUES (%s, %s, %s, CURRENT_DATE())
                """
                args = (user_id, rating, feedback_text)

            cursor.execute(insert_sql, args)
            conn.commit()

            return jsonify({
                "status": "success",
                "message": "Feedback ส่งสำเร็จ!",
                "data": {
                    "user_id": user_id,
                    "cosmetic_id": cosmetic_id,
                    "rating": rating,
                    "decision": decision
                }
            }), 200

        except mysql.connector.Error as db_err:
            if conn: conn.rollback()
            print(f"DB error on submit_feedback: {db_err}")
            return jsonify({"status": "error", "message": f"DB error: {db_err}"}), 500

        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    except Exception as e:
        print(f"submit_feedback error: {e}")
        return jsonify({"status": "error", "message": f"เกิดข้อผิดพลาดในการประมวลผล: {e}"}), 400


# --- API Endpoint: Get all Makeup Looks ---
@app.route('/ai/makeup_looks', methods=['GET'])
@jwt_required()
def get_makeup_looks():
    look_raw       = (request.args.get('category') or request.args.get('look') or request.args.get('lookCategory') or '').strip()
    q              = (request.args.get('q') or '').strip()
    undertone      = coerce_undertone(request.args.get('undertone') or request.args.get('skinTone'))
    with_cosmetics = (request.args.get('withCosmetics','false').lower() == 'true')
    limit_cos      = request.args.get('cosmeticsLimit', type=int) or 8
    debug          = request.args.get('debug') == '1'

    canon = canon_look(look_raw)
    # ถ้ารับ undertone มา → map เป็น brightness list สำหรับ suitableSkinTone
    brightness_pref = UNDERTONE_TO_BRIGHTNESS.get(undertone, ["All","Fair","Medium","Deep"])

    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)

        # ---------- LOOKS ----------
        where, params = [], []
        if canon:
            kws = LOOK_MAP[canon]["keywords"]
            name_like = " OR ".join(["LOWER(lookName) LIKE %s"]*len(kws))
            where.append("(" + name_like + ")")
            params.extend([f"%{k.lower()}%" for k in kws])
        if q:
            where.append("(LOWER(lookName) LIKE %s OR LOWER(description) LIKE %s)")
            params.extend([f"%{q.lower()}%", f"%{q.lower()}%"])

        sql_looks = "SELECT LookID, lookName, lookCategory, description FROM makeuplook"
        if where:
            sql_looks += " WHERE " + " AND ".join(where)
        sql_looks += " ORDER BY LookID ASC"
        cursor.execute(sql_looks, tuple(params))
        looks = cursor.fetchall()

        # ---------- RELATED COSMETICS (ไม่ใช้ suitableLookType เพราะว่างทั้งหมด) ----------
        related, sql_cos, params_c = [], None, None
        if with_cosmetics:
            where_c, params_c = [], []
            # ฟิลเตอร์ตาม suitableSkinTone จาก mapping (และยอมรับ All/NULL)
            where_c.append("(LOWER(COALESCE(c.suitableSkinTone,'all')) IN (" + ",".join(["%s"]*len(brightness_pref)) + "))")
            params_c.extend([b.lower() for b in brightness_pref])

            sql_cos = f"""
              SELECT c.CosmeticID, b.brandName, c.Name, c.Type,
                     COALESCE(c.ShadeCode, c.ShadeName) AS Shade,
                     c.Price, c.ImageURL, c.ProductLink,
                     c.suitableSkinTone
              FROM cosmetics c
              JOIN brand b ON b.brandID = c.BrandID
              WHERE {' AND '.join(where_c)}
              ORDER BY FIELD(c.suitableSkinTone,{','.join(['%s']*len(brightness_pref))}), c.Name ASC
              LIMIT %s
            """
            # เรียงตามลำดับความชอบ brightness แล้วค่อยตามชื่อ
            params_c.extend(brightness_pref)
            params_c.append(limit_cos)
            cursor.execute(sql_cos, tuple(params_c))
            related = cursor.fetchall()

        payload = {
            "filters": {
                "look": canon or look_raw, "q": q,
                "undertone": undertone,
                "brightnessPreference": brightness_pref,
                "withCosmetics": with_cosmetics, "cosmeticsLimit": limit_cos
            },
            "looks": looks,
            "relatedCosmetics": related
        }

        payload["data"] = looks

        if debug:
            payload["_debug"] = {"sql_looks": sql_looks, "params_looks": params,
                                 "sql_cos": sql_cos, "params_cos": params_c}
        return jsonify(payload), 200

    except Exception as e:
        print("makeup_looks error:", e)
        return jsonify({"error": "Failed to retrieve makeup looks"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


# --- NEW API Endpoint: Get all Cosmetics ---
# แก้ไขชื่อฟังก์ชัน connect_db() ถ้ามีการเปลี่ยนชื่อ
@app.route('/ai/cosmetics', methods=['GET'])
@jwt_required()
def get_all_cosmetics():
    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)
        sql_query = """
            SELECT
                c.`CosmeticID`, c.`Name`, c.`Type`, c.`Price`,
                c.`ImageURL`, c.`ProductLink`, c.`BrandID`,
                b.`brandName`,
                c.`suitableSkinTone`, c.`suitableBudgetRange`, c.`suitableLookType`,
                c.`Description`
            FROM `cosmetics` c
            JOIN `brand` b ON c.`BrandID` = b.`brandID`
            ORDER BY c.`Name` ASC
        """
        cursor.execute(sql_query)
        return jsonify(cursor.fetchall()), 200
    except mysql.connector.Error as e:
        print(f"Error fetching cosmetics: {e}")
        return jsonify({"error": "Failed to retrieve cosmetics"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


# ✅ Endpoint สำหรับ Serve รูปภาพตารางสี (ปรับ path แล้ว)
@app.route('/palettes/<filename>')
def serve_palette_image(filename):
    # 'static' คือ path ที่ชี้ไปที่โฟลเดอร์ 'static' โดยตรง
    # ตรวจสอบให้แน่ใจว่าคุณมีโฟลเดอร์ 'static' ใน root directory ของ Flask app และมีไฟล์ภาพอยู่ในนั้น
    return send_from_directory('static', filename)

@app.get("/ai/products/recommend")
@jwt_required()
def legacy_products_recommend():
    # ส่ง query string เดิมไปยัง endpoint ใหม่ (307 = keep method)
    qs = request.query_string.decode()
    target = f"/ai/cosmetics/recommendations"
    if qs:
        target += f"?{qs}"
    return redirect(target, code=307)

# --- NEW API Endpoint: Get Recommended Cosmetics based on criteria and Color Palettes ---
@app.route('/ai/cosmetics/recommendations', methods=['GET'])
@jwt_required()
def get_recommended_cosmetics():
    # ----------- รับพารามิเตอร์ -----------
    skin_tone   = (request.args.get('skinTone') or '').strip()   # ส่ง undertone เช่น "Warm Tone / Cool Tone / Neutral Tone" จะดีที่สุด
    budget_str  = (request.args.get('budget') or '').strip()
    look_raw    = (request.args.get('lookType') or request.args.get('look') or '').strip()
    style_id    = request.args.get('styleId', type=int)
    q           = (request.args.get('q') or '').strip()
    with_offers = (request.args.get('withOffers','false').lower() == 'true')

    # ----------- ช่วยแปลงช่วงงบประมาณ -----------
    def parse_budget(s: str):
        if not s: return None, None
        s = s.replace(',', '').replace('บาท','').strip()
        if '+' in s:
            try:    return int(s.split('+')[0].strip()), None
            except: return None, None
        if '-' in s:
            try:
                lo = int(s.split('-')[0].strip())
                hi = int(s.split('-')[1].strip())
                return lo, hi
            except:
                return None, None
        return None, None
    minp, maxp = parse_budget(budget_str)

    # ----------- เตรียมคำค้น look -----------
    conn = cursor = None
    look_keywords = []

    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)

        canon = canon_look(look_raw) if look_raw else None
        if canon:
            look_keywords = [k.lower() for k in LOOK_MAP[canon]["keywords"]]
        else:
            look_text = look_raw
            if not look_text and style_id:
                c2 = conn.cursor()
                c2.execute("SELECT lookName FROM makeuplook WHERE LookID=%s", (style_id,))
                r = c2.fetchone()
                c2.close()
                if r: look_text = r[0] or ''
            if look_text:
                t = look_text.lower().strip()
                toks = [t] + [x for x in t.replace('/', ' ').replace(',', ' ').split() if len(x) >= 2]
                look_keywords = list(dict.fromkeys(toks))

        # ----------- WHERE เงื่อนไขสินค้า -----------
        where, params = [], []

        # skin tone → รองรับ undertone ด้วย mapping ไปยังความสว่างที่รับได้
        brightness_pref = UNDERTONE_TO_BRIGHTNESS.get(coerce_undertone(skin_tone), None)
        if brightness_pref:
            where.append("(LOWER(COALESCE(c.`suitableSkinTone`,'all')) IN (" + ",".join(["%s"]*len(brightness_pref)) + "))")
            params.extend([b.lower() for b in brightness_pref])
        elif skin_tone:
            # ถ้าส่งมาเป็นคำอื่น ๆ ใช้ตรง ๆ + all
            where.append("(LOWER(COALESCE(c.`suitableSkinTone`,'all')) IN (%s,%s))")
            params.extend([skin_tone.lower(), "all"])

        # lookType filter (fallback: LIKE ด้วยข้อความจริง)
        if look_keywords:
            where.append("(" + " OR ".join(["LOWER(COALESCE(c.`suitableLookType`,'')) LIKE %s"]*len(look_keywords)) + ")")
            params.extend([f"%{kw}%" for kw in look_keywords])

        # keyword ค้นหา
        if q:
            where.append("(LOWER(c.`Name`) LIKE %s OR LOWER(b.`brandName`) LIKE %s OR LOWER(c.`Type`) LIKE %s)")
            params.extend([f"%{q.lower()}%", f"%{q.lower()}%", f"%{q.lower()}%"])

        # join ราคา/ดีลจาก retailer_offers (เลือก best offer)
        offer_join = """
        LEFT JOIN (
          SELECT
            oo.`CosmeticID`,
            MIN(oo.`PriceTHB`) AS `bestPrice`,
            SUBSTRING_INDEX(
              GROUP_CONCAT(oo.`URL` ORDER BY oo.`IsOfficial` DESC, oo.`Rating` DESC, oo.`PriceTHB` ASC SEPARATOR '||'),
              '||', 1
            ) AS `bestURL`,
            MAX(oo.`IsOfficial`)  AS `isOfficial`,
            MAX(oo.`Rating`)      AS `bestRating`,
            MAX(oo.`ReviewCount`) AS `bestReviews`
          FROM `retailer_offers` oo
          GROUP BY oo.`CosmeticID`
        ) o ON o.`CosmeticID` = c.`CosmeticID`
        """

        # budget range (ใช้ bestPrice ถ้ามี ไม่งั้นใช้ c.Price)
        if minp is not None:
            where.append("(COALESCE(o.`bestPrice`, c.`Price`) >= %s)"); params.append(minp)
        if maxp is not None:
            where.append("(COALESCE(o.`bestPrice`, c.`Price`) <= %s)"); params.append(maxp)

        sql = f"""
        SELECT
          c.`CosmeticID`, b.`brandName`, c.`Name`, c.`Type`,
          COALESCE(c.`ShadeCode`, c.`ShadeName`) AS Shade,
          c.`Price`, c.`ImageURL`, c.`ProductLink`,
          c.`suitableSkinTone`, c.`suitableBudgetRange`, c.`suitableLookType`,
          o.`bestPrice`, o.`bestURL`, o.`isOfficial`, o.`bestRating`, o.`bestReviews`,
          c.`Description`
        FROM `cosmetics` c
        JOIN `brand` b ON b.`brandID` = c.`BrandID`
        {offer_join}
        {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY
          o.`isOfficial` DESC,
          o.`bestRating` DESC,
          COALESCE(o.`bestPrice`, c.`Price`) ASC,
          c.`Name` ASC
        """
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        # ----------- พาเล็ตสี (ถ้าส่ง undertone มา) -----------
        palettes = []
        undertone_param = coerce_undertone(skin_tone)
        if undertone_param:
            cursor.execute("""
                SELECT `PaletteID`,`PaletteName`,`SuitableSkinTone`,`ImageURL`,`Description`
                FROM `recommendedcolorpalettes`
                WHERE LOWER(`SuitableSkinTone`) = LOWER(%s)
            """, (undertone_param,))
            palettes = cursor.fetchall()

        # ----------- ดึงผลวิเคราะห์ผิวล่าสุดของผู้ใช้ -----------
        user_id = int(get_jwt_identity())
        cursor.execute("""
          SELECT SkinTone, Undertone, Confidence
          FROM skintoneanalysis
          WHERE Users_ID=%s
          ORDER BY SkinToneAnalysisID DESC
          LIMIT 1
        """, (user_id,))
        u_skin = cursor.fetchone() or {}
        # ถ้าใน DB แยก Undertone ก็ใช้ Undertone; ถ้ายังรวมอยู่ใน SkinTone ให้ fallback จาก SkinTone
        user_undertone = (u_skin.get('Undertone') or u_skin.get('SkinTone') or '').strip()
        skin_ai_conf = (u_skin.get('Confidence') or 50) / 100.0  # 0..1, ถ้าไม่มีให้ 0.5

        # ----------- Feedback stats map (อาจยังไม่สร้าง view ก็ไม่พัง) -----------
        fb_map = {}
        try:
            cursor.execute("SELECT * FROM v_feedback_stats")
            fb_map = {r['CosmeticID']: r for r in cursor.fetchall()}
        except Exception:
            fb_map = {}

        # ----------- helper สำหรับความมั่นใจ/ที่มา -----------
        def source_trust_of(row):
            if (row.get('isOfficial') or 0) == 1:
                return 1.0
            elif (row.get('bestRating') or 0) >= 4:
                return 0.8
            return 0.6

        def conf_level(x: int) -> str:
            return "มั่นใจสูง" if x >= 75 else ("มั่นใจปานกลาง" if x >= 50 else "ควรทดลองเฉดใกล้เคียง")

        # mapping undertone → รายการความสว่างที่รับได้ (มีอยู่แล้วในไฟล์)
        brightness_pref_u = UNDERTONE_TO_BRIGHTNESS.get(user_undertone, ["All","Fair","Medium","Deep"])
        brightness_pref_lc = [b.lower() for b in brightness_pref_u]

        # ----------- คำนวณ hybrid_confidence + เหตุผล/ป้าย -----------
        augmented = []
        for r in rows:
            reasons, badges = [], []
            rule = 0.0

            # 1) ตรงกับโทนผิวผู้ใช้
            if (r.get('suitableSkinTone') or 'all').lower() in brightness_pref_lc:
                rule += 0.5
                reasons.append("โทนเฉดเข้ากับผลวิเคราะห์ผิวของคุณ")

            # 2) แหล่งทางการ / ข้อมูลครบ
            if (r.get('isOfficial') or 0) == 1:
                rule += 0.2
                badges.append("official")
                reasons.append("มีร้านทางการ (Official) ให้เลือก")

            if r.get('ImageURL'):
                rule += 0.05
            if r.get('Description'):
                rule += 0.05

            # Cap rule_confidence ที่ 1.0
            rule_conf = min(1.0, rule)

            # 3) ที่มาของข้อมูล
            s_trust = source_trust_of(r)

            # 4) crowd boost จาก feedback
            st = fb_map.get(r['CosmeticID'])
            if st and (st.get('total_reviews') or 0) >= 5:
                liked = st.get('liked') or 0
                disliked = st.get('disliked') or 0
                total = st.get('total_reviews') or 1
                net = (liked - disliked) / float(total)
                crowd_boost = max(0.0, min(1.0, (net + 1.0) / 2.0))
                if liked > disliked:
                    reasons.append("ผู้ใช้ส่วนใหญ่ให้คะแนนเชิงบวก")
            else:
                crowd_boost = 0.5

            # ----------- hybrid_confidence -----------
            hybrid = int(round(100 * (
                0.55 * rule_conf +
                0.20 * skin_ai_conf +
                0.15 * s_trust +
                0.10 * crowd_boost
            )))
            level = conf_level(hybrid)

            augmented.append({
                **r,
                "hybrid_confidence": hybrid,
                "confidence_level": level,
                "badges": list(dict.fromkeys(badges)),
                "reasons": reasons[:3]  # แสดง 2-3 ข้อก็พอ
            })

        # เรียงตามความมั่นใจสูง → ต่ำ
        augmented.sort(key=lambda x: x['hybrid_confidence'], reverse=True)

        return jsonify({
            "recommendedCosmetics": augmented,
            "recommendedColorPalettes": palettes
        }), 200

    except Exception as e:
        print("recommendations error:", e)
        return jsonify({"error": "Failed to retrieve recommendations"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# --- ดึงรายการข้อเสนอ (offers) ของสินค้า 1 ชิ้น จากตาราง retailer_offers
@app.route('/ai/cosmetics/<int:cosmetic_id>', methods=['GET'])
@jwt_required()
def get_cosmetic_detail(cosmetic_id):
    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT c.`CosmeticID`, b.`brandName`, c.`Name`, c.`Type`,
                   COALESCE(c.`ShadeCode`, c.`ShadeName`) AS Shade,
                   c.`Price`, c.`ImageURL`, c.`ProductLink`,
                   c.`suitableSkinTone`, c.`suitableBudgetRange`,
                   c.`suitableLookType`, c.`Description`
            FROM `cosmetics` c
            JOIN `brand` b ON b.`brandID` = c.`BrandID`
            WHERE c.`CosmeticID`=%s
        """, (cosmetic_id,))
        item = cursor.fetchone()
        if not item:
            return jsonify({"error": "Cosmetic not found"}), 404

        # best offer (ถ้ามี)
        cursor.execute("""
            SELECT `Retailer`,`URL`,`ImageURL`,`PriceTHB`,`Rating`,`ReviewCount`,`IsOfficial`,`LastUpdate`
            FROM `retailer_offers`
            WHERE `CosmeticID`=%s
            ORDER BY `IsOfficial` DESC, `Rating` DESC, `PriceTHB` ASC
            LIMIT 1
        """, (cosmetic_id,))
        best_offer = cursor.fetchone()

        return jsonify({"item": item, "bestOffer": best_offer}), 200
    except Exception as e:
        print("detail error:", e)
        return jsonify({"error": "Failed to retrieve cosmetic detail"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/ai/brands', methods=['GET'])
@jwt_required()
def list_brands():
    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT brandID, brandName FROM brand ORDER BY brandName ASC")
        return jsonify(cursor.fetchall()), 200
    except Exception as e:
        print("brands error:", e)
        return jsonify({"error": "Failed to retrieve brands"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()



# ✅ รัน Flask API
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=False, threaded=True)
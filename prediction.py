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
                        INSERT INTO similarity (similarityDetail_Percent, ThaiCelebrities_ID, User_ID, similarity_Date)
                        VALUES (%s, %s, %s, CURRENT_DATE)
                    '''
                    cursor.execute(insert_query, (
                        best_match['confidence'],
                        celeb_id,
                        current_user_id
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

# ✅ API ทำนายสีผิว
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
    file.save(file_path)

    predicted_class, confidence, all_probabilities = predict_skin_tone_from_image(file_path)
    os.remove(file_path)

    if predicted_class is None:
        return jsonify({"error": "Failed to predict skin tone"}), 500

    brightness_tone = map_class_to_brightness_tone(predicted_class)
    overall_undertone, undertone_percentages = calculate_overall_undertone(all_probabilities)

    try:
        current_user_id = get_jwt_identity()
        conn = connect_db()
        if conn is None:
            print("❌ Error: Database connection failed for skin tone API")
        else:
            cursor = conn.cursor()
            insert_query = """
            INSERT INTO SkinToneAnalysis
            (SkinTone, Users_ID)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (
                overall_undertone, # เก็บ overall_undertone (โทนอุ่น/เย็น/กลาง)
                current_user_id
            ))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ บันทึกข้อมูล SkinTone ลงฐานข้อมูลสำเร็จ - Users_ID: {current_user_id}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึกข้อมูล SkinTone ลงฐานข้อมูล: {e}")

    return jsonify({
        "overall_undertone": overall_undertone,
        "predicted_class": predicted_class,
        "confidence": confidence, # เพิ่มความมั่นใจของคลาสที่ทำนาย
        "all_probabilities": all_probabilities, # เพิ่มความน่าจะเป็นทั้งหมด
        "brightness_tone": brightness_tone, # เพิ่มโทนความสว่าง
        "undertone_percentages": undertone_percentages, # เพิ่มเปอร์เซ็นต์ของ undertone ทั้งหมด
        "message": "Skin tone prediction successful."
    }), 200

# ✅ API รับ Feedback จาก Android App (แก้ไขแล้ว)
@app.route('/ai/submit_feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    try:
        data = request.get_json()
        print(f"Received JSON data for feedback: {data}")

        if not data:
            print("Error: No JSON data received in feedback request.")
            return jsonify({"message": "JSON Data ไม่ถูกต้อง", "status": "error"}), 400

        rating = data.get('rating')
        feedback_text = data.get('feedback_text')

        if rating is None or feedback_text is None:
            print(f"Error: Missing required fields. Rating: {rating}, Feedback Text: {feedback_text}")
            return jsonify({"message": "ข้อมูล Feedback ไม่ครบถ้วน (ต้องมี rating และ feedback_text)", "status": "error"}), 400

        # ตรวจสอบประเภทข้อมูล (Optional แต่ดีกว่า)
        if not isinstance(rating, int): # ✅ ตรวจสอบว่าเป็น int หรือไม่
            # ถ้า Android ส่ง float (เช่น 3.0) ให้แปลงเป็น int
            if isinstance(rating, float):
                rating = int(rating)
            else:
                print(f"Error: Rating is not an integer. Type: {type(rating)}, Value: {rating}")
                return jsonify({"message": "Rating ต้องเป็นตัวเลขจำนวนเต็ม", "status": "error"}), 400

        if not isinstance(feedback_text, str):
            print(f"Error: Feedback text is not a string. Type: {type(feedback_text)}, Value: {feedback_text}")
            return jsonify({"message": "Feedback text ต้องเป็นข้อความ", "status": "error"}), 400

        # ✅ ดึง User ID จาก JWT Token
        current_user_id = get_jwt_identity()

        conn = None
        cursor = None
        try:
            conn = connect_db()
            if conn is None:
                print("❌ Error: Database connection failed for submit_feedback API")
                return jsonify({"message": "Database connection failed", "status": "error"}), 500

            cursor = conn.cursor()

            # ✅ นี่คือส่วนสำคัญ: INSERT Data ลงในตาราง feedback
            # สมมติว่าตาราง feedback ของคุณมีคอลัมน์เช่น
            # Feedback_ID (PK, AUTO_INCREMENT)
            # Users_ID (FK, จาก JWT)
            # rating (INT)
            # feedback_text (TEXT)
            # submission_date (DATE)
            # ถ้าชื่อคอลัมน์ไม่ตรง ให้ปรับแก้ตามชื่อคอลัมน์ในฐานข้อมูลของคุณ
            insert_query = """
                INSERT INTO feedback (UserID, Rating, CommentText, Date)
                VALUES (%s, %s, %s, CURRENT_DATE())
            """
            cursor.execute(insert_query, (current_user_id, rating, feedback_text))
            conn.commit() # ✅ สำคัญมาก: commit การเปลี่ยนแปลงลงฐานข้อมูล

            print(f"✅ บันทึก Feedback ลงฐานข้อมูลสำเร็จ: User ID={current_user_id}, Rating={rating}, Feedback='{feedback_text}'")
            return jsonify({"message": "Feedback ส่งสำเร็จ!", "status": "success"}), 200

        except mysql.connector.Error as db_err:
            print(f"❌ Error saving feedback to DB: {db_err}")
            # ถ้าเกิด error จาก DB ให้ rollback เพื่อไม่ให้เกิดปัญหาค้าง
            if conn:
                conn.rollback()
            return jsonify({"message": f"เกิดข้อผิดพลาดในการบันทึก Feedback ลงฐานข้อมูล: {db_err}", "status": "error"}), 500
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิดในการประมวลผล Feedback: {e}")
            return jsonify({"message": f"เกิดข้อผิดพลาดที่ Server: {e}", "status": "error"}), 500
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    except Exception as e: # Catch exception จาก request.get_json() หรือ data.get()
        print(f"❌ Error processing feedback request body: {e}")
        return jsonify({"message": f"เกิดข้อผิดพลาดในการอ่านข้อมูล: {e}", "status": "error"}), 400


# --- API Endpoint: Get all Makeup Looks ---
@app.route('/ai/makeup_looks', methods=['GET'])
def get_makeup_looks():
    conn = None
    cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT LookID, lookName, lookCategory, description FROM MakeupLook ORDER BY lookName ASC")
        looks = cursor.fetchall()

        return jsonify({
            "status": "success",
            "data": looks,
            "count": len(looks)
        }), 200
    except Exception as e:
        print(f"Error fetching makeup looks: {e}")
        return jsonify({"error": "Failed to retrieve makeup looks"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- NEW API Endpoint: Get all Cosmetics ---
# แก้ไขชื่อฟังก์ชัน connect_db() ถ้ามีการเปลี่ยนชื่อ
@app.route('/ai/cosmetics', methods=['GET'])
def get_all_cosmetics():
    conn = None
    cursor = None
    try:
        conn = connect_db() # ใช้ connect_db() ที่คุณกำหนดไว้
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor(dictionary=True)
        sql_query = """
            SELECT
                c.CosmeticID,
                c.Name,
                c.Type,
                c.Price,
                c.ImageURL,
                c.ProductLink,
                c.BrandID,
                b.brandName, -- ดึงชื่อแบรนด์มาด้วย
                c.suitableSkinTone,
                c.suitableBudgetRange,
                c.suitableLookType
            FROM
                Cosmetics c
            JOIN
                Brand b ON c.BrandID = b.brandID
            ORDER BY
                c.Name ASC
        """
        cursor.execute(sql_query)
        cosmetics = cursor.fetchall()

        return jsonify(cosmetics)
    except mysql.connector.Error as e: # เปลี่ยนจาก Error เป็น mysql.connector.Error
        print(f"Error fetching cosmetics: {e}")
        return jsonify({"error": "Failed to retrieve cosmetics"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ✅ Endpoint สำหรับ Serve รูปภาพตารางสี (ปรับ path แล้ว)
@app.route('/palettes/<filename>')
def serve_palette_image(filename):
    # 'static' คือ path ที่ชี้ไปที่โฟลเดอร์ 'static' โดยตรง
    # ตรวจสอบให้แน่ใจว่าคุณมีโฟลเดอร์ 'static' ใน root directory ของ Flask app และมีไฟล์ภาพอยู่ในนั้น
    return send_from_directory('static', filename)


# --- NEW API Endpoint: Get Recommended Cosmetics based on criteria and Color Palettes ---
@app.route('/ai/cosmetics/recommendations', methods=['GET'])
# @jwt_required() # <--- Un-comment ถ้าต้องการให้ API นี้ต้อง Login ก่อน
def get_recommended_cosmetics():
    skin_tone = request.args.get('skinTone')
    budget_range = request.args.get('budget')
    look_type = request.args.get('lookType')

    if not skin_tone:
        return jsonify({"error": "skinTone parameter is required"}), 400

    conn = None
    cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor(dictionary=True)

        # --- 1. Fetch Recommended Cosmetics (filter ตาม skinTone, budget, lookType) ---
        cosmetic_where_clauses = ["(c.suitableSkinTone = %s OR c.suitableSkinTone = 'All')"]
        cosmetic_params = [skin_tone]

        if budget_range:
            cosmetic_where_clauses.append("c.suitableBudgetRange = %s")
            cosmetic_params.append(budget_range)

        if look_type:
            cosmetic_where_clauses.append("c.suitableLookType = %s")
            cosmetic_params.append(look_type)

        sql_cosmetics_query = f"""
            SELECT
                c.CosmeticID, c.Name, c.ShadeCode, c.ShadeName, c.Type, c.Price, c.ImageURL, c.ProductLink, b.brandName,
                c.suitableSkinTone, c.suitableBudgetRange, c.suitableLookType
            FROM
                Cosmetics c
            JOIN
                Brand b ON c.BrandID = b.brandID
            WHERE
                {' AND '.join(cosmetic_where_clauses)}
            ORDER BY
                c.Name ASC
        """
        cursor.execute(sql_cosmetics_query, tuple(cosmetic_params))
        recommended_cosmetics = cursor.fetchall()

        # --- 2. Fetch Recommended Color Palettes (ดึงรูปตารางสีตาม skinTone เท่านั้น) ---
        sql_palettes_query = """
            SELECT PaletteID, PaletteName, SuitableSkinTone, ImageURL, Description
            FROM RecommendedColorPalettes
            WHERE SuitableSkinTone = %s;
        """
        cursor.execute(sql_palettes_query, (skin_tone,))
        recommended_color_palettes = cursor.fetchall()

        # --- Combine and Return ---
        return jsonify({
            "recommendedCosmetics": recommended_cosmetics,
            "recommendedColorPalettes": recommended_color_palettes
        })

    except mysql.connector.Error as e: # เปลี่ยนจาก Error เป็น mysql.connector.Error
        print(f"Error fetching recommendations: {e}")
        return jsonify({"error": "Failed to retrieve recommendations"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ✅ รัน Flask API
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=False, threaded=True)
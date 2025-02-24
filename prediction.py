import mysql.connector
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import get_custom_objects
from datetime import datetime

# ✅ ตั้งค่าการเชื่อมต่อ MySQL
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "db_miniprojectfinal"
}

# ✅ สร้าง Flask App
app = Flask(__name__)
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

# ✅ โหลดโมเดล
MODEL_PATH = "best_model_efficientnetB3.keras"
try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"swish": tf.keras.activations.swish, "FixedDropout": FixedDropout, "AdamW": AdamW}
    )
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_input, verbose=0)
    feature_extractor = tf.keras.Model(inputs=model.inputs[0], outputs=model.get_layer("global_average_pooling2d").output)
    print("✅ โหลดโมเดลสำเร็จ!")
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

# ✅ ฟังก์ชันเชื่อมต่อ MySQL
def connect_db():
    return mysql.connector.connect(**db_config)

# ✅ ฟังก์ชันเตรียมภาพ
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img_resized = cv2.resize(img, (1024, 1024))
        img_resized = cv2.resize(img_resized, target_size)
        img_resized = img_to_array(img_resized) / 255.0
        return np.expand_dims(img_resized, axis=0)
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดหรือปรับขนาดภาพ {image_path}: {e}")
        return None

# ✅ ฟังก์ชันดึง Feature Vector
def get_feature_vector(image_path):
    img = preprocess_image(image_path)
    return feature_extractor.predict(img, verbose=0) if img is not None else None

# ✅ ฟังก์ชันค้นหาใบหน้าที่คล้ายที่สุด
def find_most_similar_face(test_vector):
    if feature_database is None or label_database is None:
        return None, None
    similarities = cosine_similarity(test_vector.reshape(1, -1), feature_database)
    best_match_idx = np.argmax(similarities)
    best_match_label = label_database[best_match_idx]
    confidence = similarities[0][best_match_idx] * 100
    return best_match_label, round(float(confidence), 2)

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json(silent=True)

    print("📩 Received Data:", data)  # ✅ Debug JSON ที่ได้รับ
    
    if not data or "username" not in data or "password" not in data:
        return jsonify({"status": "error", "message": "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"}), 400

    username = data["username"]
    password = data["password"]
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({"status": "error", "message": "ชื่อผู้ใช้นี้ถูกใช้ไปแล้ว"}), 400

        # กำหนด Role_ID = 1 (User) ในการสมัคร
        cursor.execute("INSERT INTO users (username, password, Role_ID) VALUES (%s, %s, 1)", (username, hashed_password))
        conn.commit()

        # ดึงข้อมูล Role_ID ของผู้ใช้ใหม่
        cursor.execute("SELECT Role_ID FROM users WHERE username = %s", (username,))
        role_id = cursor.fetchone()[0]  # ดึง Role_ID จากฐานข้อมูล

        # สร้าง JWT Token ด้วย Role_ID ที่ถูกต้อง
        access_token = create_access_token(identity={"username": username, "role": role_id})

        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "สมัครสมาชิกสำเร็จ!",
            "token": access_token
        }), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ✅ API `/auth/login` - เข้าสู่ระบบ
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"}), 400

    try:
        conn = connect_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            # สร้าง JWT Token โดยเพิ่ม user_id ลงใน payload
            access_token = create_access_token(identity={"username": user["username"], "user_id": user["Users_ID"], "role": user["Role_ID"]})
            return jsonify({"message": "เข้าสู่ระบบสำเร็จ!", "token": access_token}), 200
        else:
            return jsonify({"error": "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ API `/ai/predict` - ทำนายใบหน้าและบันทึกลงฐานข้อมูล

@app.route('/ai/predict', methods=['POST'])
@jwt_required()  # ตรวจสอบว่าเป็นผู้ใช้ที่เข้าสู่ระบบแล้ว
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    test_face_vector = get_feature_vector(file_path)
    if test_face_vector is None:
        os.remove(file_path)
        return jsonify({"error": "Failed to process image"}), 500

    best_match_name, confidence = find_most_similar_face(test_face_vector)
    os.remove(file_path)

    if best_match_name:
        try:
            conn = connect_db()
            cursor = conn.cursor()

            cursor.execute("SELECT ThaiCelebrities_ID FROM thaicelebrities WHERE ThaiCelebrities_name = %s", (best_match_name,))
            result = cursor.fetchone()

            if result:
                celebrity_id = result[0]
                similarity_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # ดึง User_ID จาก JWT Token
                current_user = get_jwt_identity()  # ดึงข้อมูลจาก JWT token
                print(current_user)  # พิมพ์ค่า user_id

                if "user_id" not in current_user:
                    return jsonify({"error": "user_id not found in token"}), 400  # ถ้าไม่มี user_id ใน token

                user_id = current_user["user_id"]  # ดึง user_id จาก token

                # บันทึกข้อมูลในตาราง similarity พร้อม User_ID
                cursor.execute(
                    "INSERT INTO similarity (similarity_Date, similarityDetail_Percent, ThaiCelebrities_ID, user_id) VALUES (%s, %s, %s, %s)",
                    (similarity_date, confidence, celebrity_id, user_id)  # เพิ่ม user_id
                )
                conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึกข้อมูลลง Database: {e}")

        return jsonify({
            "predicted_match": best_match_name,
            "confidence_score": confidence
        }), 200
    else:
        return jsonify({"error": "No matching face found"}), 500


# ✅ รัน Flask API
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

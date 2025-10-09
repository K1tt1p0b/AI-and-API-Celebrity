# app.py
import os
import cv2
import json
import math
import random
import numpy as np
import mysql.connector
import tensorflow as tf

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import get_custom_objects
from PIL import Image

# PyTorch (skin tone model)
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# -----------------------------
# Basic setup
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU for TF/PyTorch

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "db_miniprojectfinal"
}

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)
app.config["JWT_SECRET_KEY"] = "ggygyuf6ydfyh8u5yusfuy"
jwt = JWTManager(app)

# upload temp dir (for incoming files)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------- NEW: static product images dir + route ----------
# เก็บไฟล์รูปสินค้าจริงไว้ที่ static/products และเสิร์ฟผ่าน /products/<filename>
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_DIR = os.path.join(BASE_DIR, "static", "products")
os.makedirs(PRODUCTS_DIR, exist_ok=True)

@app.route("/products/<path:filename>")
def serve_product_image(filename):
    # ตัวอย่าง URL: http://<host>:5003/products/1717234_abc.jpg
    # ใน DB เก็บ ImageURL เป็น 'products/1717234_abc.jpg'
    resp = send_from_directory(PRODUCTS_DIR, filename)
    try:
        # optional cache: 1 วัน
        resp.cache_control.max_age = 24 * 60 * 60
    except Exception:
        pass
    return resp
# ----------------------------------------------------------

# -----------------------------
# Keras custom layers
# -----------------------------
get_custom_objects().update({"swish": tf.keras.activations.swish})

class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

get_custom_objects().update({"FixedDropout": FixedDropout})

# -----------------------------
# Face feature extractor (ResNet50)
# -----------------------------
MODEL_PATH = "resnet50_final_model_v2_edit_1.keras"
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"swish": tf.keras.activations.swish, "FixedDropout": FixedDropout, "AdamW": AdamW}
    )
    print("✅ Loaded ResNet50 model")

    resnet_base = model.get_layer("resnet50")
    resnet_base.trainable = False

    feature_extractor = tf.keras.Sequential([
        resnet_base,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    _ = feature_extractor.predict(np.zeros((1, 224, 224, 3)), verbose=0)
except Exception as e:
    print(f"❌ Failed to load ResNet50 feature extractor: {e}")
    model = None
    feature_extractor = None

# -----------------------------
# Face feature DB
# -----------------------------
FEATURE_DB_PATH = "feature_database.npy"
LABELS_DB_PATH  = "label_database.npy"
try:
    feature_database = np.load(FEATURE_DB_PATH, allow_pickle=True)
    label_database   = np.load(LABELS_DB_PATH, allow_pickle=True)
    print(f"✅ Feature DB loaded: {len(label_database)} labels")
except Exception as e:
    print(f"❌ Feature DB load error: {e}")
    feature_database = None
    label_database = None

# -----------------------------
# Skin tone model (PyTorch, 4 classes)
# -----------------------------
skin_tone_model = None
SKIN_TONE_MODEL_PATH = "mobilenet_v2_skintonemodel.pth"
try:
    mobilenet_v2_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    num_skin_tone_classes = 4  # deep dark, fair, brown, medium
    mobilenet_v2_model.classifier[1] = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Linear(mobilenet_v2_model.classifier[1].in_features, num_skin_tone_classes)
    )
    skin_tone_model = mobilenet_v2_model
    skin_tone_model.load_state_dict(torch.load(SKIN_TONE_MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    skin_tone_model.eval()
    print(f"✅ Skin tone model loaded from {SKIN_TONE_MODEL_PATH}")
except ImportError:
    print("❌ PyTorch/torchvision missing: pip install torch torchvision")
    skin_tone_model = None
except Exception as e:
    print(f"❌ Cannot load skin tone model: {e}")
    skin_tone_model = None

# -----------------------------
# DB helpers
# -----------------------------
def connect_db():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        print(f"❌ DB connect error: {err}")
        return None

# -----------------------------
# Image helpers (face similarity)
# -----------------------------
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        img = img_to_array(img)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"❌ preprocess_image error: {e}")
        return None

def get_feature_vector(image_path):
    img = preprocess_image(image_path)
    if img is not None and feature_extractor is not None:
        return feature_extractor.predict(img, verbose=0)[0]
    return None

def find_top_similar_faces(test_vector, top_n=5):
    if feature_database is None or label_database is None:
        return []
    sims = cosine_similarity(test_vector.reshape(1, -1), feature_database)[0]
    idxs = sims.argsort()[::-1]
    results, seen = [], set()
    for i in idxs:
        label = label_database[i]
        score = round(float(sims[i]) * 100, 2)
        if label not in seen:
            results.append({"name": label, "confidence": score, "raw_index": int(i)})
            seen.add(label)
        if len(results) == top_n:
            break
    return results

# -----------------------------
# Skin tone classification helpers
# -----------------------------
RAW2BRIGHT = {
    "fair": "Fair",
    "medium": "Medium",
    "brown": "Brown",
    "deep dark": "Deep"
}

def map_raw_to_brightness(raw_cls: str) -> str:
    return RAW2BRIGHT.get((raw_cls or "").lower(), "Medium")

def brightness_label_th(en: str):
    th = {"Fair":"โทนสว่าง", "Medium":"โทนกลาง", "Brown":"โทนน้ำตาล", "Deep":"โทนเข้ม"}
    return th.get(en, "ไม่ระบุโทน")

def predict_skin_tone_from_image(image_path):
    classes = ["deep dark", "fair", "brown", "medium"]
    if skin_tone_model is None:
        rnd = [random.random() for _ in classes]
        s = sum(rnd)
        probs = [p/s for p in rnd]
        ap = {c: round(100*p, 2) for c, p in zip(classes, probs)}
        raw = max(ap, key=ap.get)
        return raw, ap[raw], ap
    try:
        img = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = skin_tone_model(input_tensor)
        prob = torch.nn.functional.softmax(out[0], dim=0).numpy() * 100.0
        ap = {classes[i]: round(float(prob[i]), 2) for i in range(len(classes))}
        raw = max(ap, key=ap.get)
        return raw, ap[raw], ap
    except Exception as e:
        print(f"❌ PyTorch predict error: {e}")
        return None, None, None

# -----------------------------
# ITA / Lab utilities
# -----------------------------
def simple_skin_mask_ycrcb(bgr_img: np.ndarray):
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    return mask

def image_area_to_lab_stats_bgr(bgr_img: np.ndarray, skin_mask: np.ndarray = None):
    if skin_mask is not None:
        bgr = bgr_img[skin_mask > 0]
        if bgr.size == 0:
            return None
        bgr = bgr.reshape(-1, 1, 3)
    else:
        bgr = bgr_img
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0] * (100.0 / 255.0)
    a = lab[:, 1] - 128.0
    b = lab[:, 2] - 128.0
    return float(L.mean()), float(a.mean()), float(b.mean())

def compute_ita_from_lab(L_star: float, b_star: float) -> float:
    if abs(b_star) < 1e-6:
        b_star = 1e-6
    return math.degrees(math.atan((L_star - 50.0) / b_star))

def ita_to_bucket(ita_deg: float) -> str:
    if ita_deg > 55:
        return "Fair"
    elif 28 < ita_deg <= 55:
        return "Medium"
    elif 10 < ita_deg <= 28:
        return "Brown"
    else:
        return "Deep"

def predict_brightness_via_ita(image_path: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    skin = simple_skin_mask_ycrcb(bgr)
    lab_stats = image_area_to_lab_stats_bgr(bgr, skin)
    if lab_stats is None:
        return None
    L_star, a_star, b_star = lab_stats
    ita_deg = compute_ita_from_lab(L_star, b_star)
    return {
        "L*": round(L_star, 2),
        "a*": round(a_star, 2),
        "b*": round(b_star, 2),
        "ITA_deg": round(ita_deg, 2),
        "brightness": ita_to_bucket(ita_deg)
    }

# -----------------------------
# CIEDE2000 ΔE00 (no dependency)
# -----------------------------
def delta_e_ciede2000(lab1, lab2):
    import math
    L1, a1, b1 = lab1; L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2.0
    C1 = math.hypot(a1, b1); C2 = math.hypot(a2, b2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt((avg_C**7) / (avg_C**7 + 25**7))) if avg_C != 0 else 0
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1); C2p = math.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2.0
    def hp(a, b):
        h = math.degrees(math.atan2(b, a))
        return h + 360 if h < 0 else h
    h1p = 0 if C1p == 0 else hp(a1p, b1)
    h2p = 0 if C2p == 0 else hp(a2p, b2)
    if abs(h1p - h2p) > 180:
        avg_hp = (h1p + h2p + 360) / 2.0
    else:
        avg_hp = (h1p + h2p) / 2.0
    T = (1 - 0.17 * math.cos(math.radians(avg_hp - 30))
           + 0.24 * math.cos(math.radians(2 * avg_hp))
           + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
           - 0.20 * math.cos(math.radians(4 * avg_hp - 63)))
    dhp = h2p - h1p
    if abs(dhp) > 180:
        dhp -= 360 if dhp > 0 else -360
    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))
    Sl = 1 + (0.015 * (avg_L - 50) ** 2) / math.sqrt(20 + (avg_L - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    delta_ro = 30 * math.exp(-(((avg_hp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt((avg_Cp ** 7) / ((avg_Cp ** 7) + (25 ** 7)))
    Rt = -math.sin(math.radians(2 * delta_ro)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))

# -----------------------------
# Helpers
# -----------------------------
def canon_suitable_tone(val: str):
    if not val:
        return "Universal"
    v = val.strip().lower()
    if v in {"universal", "all", "ทั้งหมด", "ทุกโทน"}:
        return "Universal"
    m = {
        "fair":"Fair","light":"Fair",
        "medium":"Medium",
        "brown":"Brown",
        "deep":"Deep","deep dark":"Deep"
    }
    return m.get(v, val)

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

# CSV-safe LIKE matcher for tone lists like "Light, Medium, Brown"
def tone_csv_like_clause(column_alias="c.`suitableSkinTone`", tone_value="fair"):
    # LOCATE(',fair,', CONCAT(',', lower(col), ',')) > 0
    return f"LOCATE(%s, CONCAT(',', LOWER(COALESCE({column_alias},'')), ',')) > 0", f",{tone_value.lower()},"

# -----------------------------
# (NEW) Static palette mapping for /ai/cosmetics/recommendations
# -----------------------------
PALETTE_BY_TONE = {
    "Fair":   "fair.jpg",
    "Medium": "medium.jpg",
    "Brown":  "brown.jpg",
    "Deep":   "deep.jpg",
}

# -----------------------------
# AUTH
# -----------------------------
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
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username=%s", (username,))
        if cur.fetchone():
            return jsonify({"status": "error", "message": "ชื่อผู้ใช้นี้ถูกใช้ไปแล้ว"}), 400
        cur.execute("INSERT INTO users (username, password, Role_ID) VALUES (%s, %s, 1)", (username, hashed_password))
        conn.commit(); cur.close(); conn.close()
        return jsonify({"status": "success", "message": "สมัครสมาชิกสำเร็จ!"}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/ai/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username"); password = data.get("password")
    if not username or not password:
        return jsonify({"error": "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"}), 400
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        cur.close(); conn.close()
        if user and bcrypt.check_password_hash(user["password"], password):
            token = create_access_token(identity=str(user["Users_ID"]))
            return jsonify({"message":"เข้าสู่ระบบสำเร็จ!", "token": token}), 200
        return jsonify({"error": "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# FACE PREDICT
# -----------------------------
@app.route('/ai/predict', methods=['POST'])
@jwt_required()
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    fname = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(path)

    vec = get_feature_vector(path)
    try: os.remove(path)
    except Exception: pass

    if vec is None:
        return jsonify({"error": "Failed to process image"}), 500

    top_matches = find_top_similar_faces(vec, top_n=5)

    if top_matches:
        best = top_matches[0]
        try:
            uid = get_jwt_identity()
            conn = connect_db()
            if conn:
                cur = conn.cursor()
                cur.execute("SELECT ThaiCelebrities_ID FROM thaicelebrities WHERE ThaiCelebrities_name=%s", (best['name'],))
                row = cur.fetchone()
                celeb_id = row[0] if row else None
                if celeb_id is not None:
                    cur.execute("""
                        INSERT INTO similarity (similarityDetail_Percent, ThaiCelebrities_ID, Users_ID, similarity_Date)
                        VALUES (%s, %s, %s, CURRENT_DATE)
                    """, (best['confidence'], celeb_id, int(uid)))
                    conn.commit()
                cur.close(); conn.close()
        except Exception as e:
            print(f"❌ Save similarity error: {e}")

    return jsonify({
        "top_matches": [{"name": m["name"], "confidence": m["confidence"]} for m in top_matches],
        "message": "Top 5 most similar celebrities found."
    }), 200 if top_matches else 500

# -----------------------------
# SKIN TONE PREDICT (AI + ITA fusion)
# -----------------------------
@app.route('/ai/predict_skin_tone', methods=['POST'])
@jwt_required()
def predict_skin_tone():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided for skin tone prediction"}), 400
    file = request.files['image']
    fname = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    try:
        file.save(path)

        raw_class, ai_conf, ai_all = predict_skin_tone_from_image(path)
        ai_brightness = map_raw_to_brightness(raw_class) if raw_class else None

        ita_res = predict_brightness_via_ita(path)
        ita_brightness = ita_res["brightness"] if ita_res else None

        final_brightness = None; reasons = []
        if ai_brightness and ita_brightness:
            if ai_brightness == ita_brightness:
                final_brightness = ai_brightness; reasons.append("AI และ ITA ให้ผลสอดคล้องกัน")
            else:
                if (ai_conf or 0) >= 70:
                    final_brightness = ai_brightness; reasons.append("AI มั่นใจสูง ใช้ผล AI")
                else:
                    final_brightness = ita_brightness; reasons.append("AI มั่นใจไม่สูง ใช้ผล ITA")
        elif ai_brightness:
            final_brightness = ai_brightness; reasons.append("มีเฉพาะผล AI")
        elif ita_brightness:
            final_brightness = ita_brightness; reasons.append("มีเฉพาะผล ITA")
        else:
            return jsonify({"error":"Cannot determine brightness class"}), 500

        brightness_th = brightness_label_th(final_brightness)

        # save to DB (SkinTone + ITA metrics if columns exist)
        uid = int(get_jwt_identity())
        conn = connect_db()
        if conn:
            cur = conn.cursor()
            has_ita = has_L = has_b = False
            try:
                cur.execute("SHOW COLUMNS FROM skintoneanalysis LIKE 'ITA_Deg'"); has_ita = cur.fetchone() is not None
                cur.execute("SHOW COLUMNS FROM skintoneanalysis LIKE 'L_star'");   has_L  = cur.fetchone() is not None
                cur.execute("SHOW COLUMNS FROM skintoneanalysis LIKE 'b_star'");   has_b  = cur.fetchone() is not None
            except: pass

            if has_ita or has_L or has_b:
                cur.execute(f"""
                    INSERT INTO skintoneanalysis (SkinTone, Undertone, Confidence, Users_ID
                        {", ITA_Deg" if has_ita else ""}{", L_star" if has_L else ""}{", b_star" if has_b else ""})
                    VALUES (%s, %s, %s, %s
                        {", %s" if has_ita else ""}{", %s" if has_L else ""}{", %s" if has_b else ""})
                """, tuple([
                    final_brightness, None, float(ai_conf or 0), uid
                ] + ([ita_res["ITA_deg"]] if (has_ita and ita_res) else [])
                  + ([ita_res["L*"]] if (has_L and ita_res) else [])
                  + ([ita_res["b*"]] if (has_b and ita_res) else [])))
            else:
                cur.execute("""
                    INSERT INTO skintoneanalysis (SkinTone, Undertone, Confidence, Users_ID)
                    VALUES (%s, %s, %s, %s)
                """, (final_brightness, None, float(ai_conf or 0), uid))
            conn.commit(); cur.close(); conn.close()

        # NOTE: fields align with Android PrizeActivity (old schema too)
        return jsonify({
            "brightness_class": final_brightness,
            "brightness_label_th": brightness_th,
            "confidence": float(ai_conf or 0),
            "ai": {"raw_class": raw_class, "brightness": ai_brightness, "confidence": ai_conf, "probs": ai_all},
            "ita": ita_res,
            "agreement": (ai_brightness == ita_brightness) if (ai_brightness and ita_brightness) else None,
            "explanations": reasons,
            "message": "Fused AI + ITA brightness classification."
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(path): os.remove(path)
        except Exception:
            pass

# -----------------------------
# FEEDBACK
# -----------------------------
@app.route('/ai/submit_feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "JSON ไม่ถูกต้อง"}), 400
        rating = data.get('rating', None)
        feedback_text = data.get('feedback_text', "")
        cosmetic_id = data.get('cosmetic_id', None)

        if rating is None:
            return jsonify({"status": "error", "message": "ต้องมี rating"}), 400
        if isinstance(rating, float): rating = int(rating)
        if not isinstance(rating, int):
            return jsonify({"status": "error", "message": "rating ต้องเป็นจำนวนเต็ม"}), 400
        if rating < 1 or rating > 5:
            return jsonify({"status": "error", "message": "rating ต้องอยู่ระหว่าง 1–5"}), 400

        if not isinstance(feedback_text, str):
            return jsonify({"status": "error", "message": "feedback_text ต้องเป็นข้อความ"}), 400
        feedback_text = feedback_text.strip()
        if len(feedback_text) > 1000: feedback_text = feedback_text[:1000]

        if cosmetic_id is not None:
            try:
                cosmetic_id = int(cosmetic_id)
                if cosmetic_id <= 0: cosmetic_id = None
            except Exception:
                cosmetic_id = None

        uid = int(get_jwt_identity())
        conn = cursor = None
        try:
            conn = connect_db()
            if conn is None:
                return jsonify({"status": "error", "message": "Database connection failed"}), 500
            cursor = conn.cursor()
            cursor.execute("SHOW COLUMNS FROM feedback LIKE 'CosmeticID'")
            has_cos_col = cursor.fetchone() is not None

            if has_cos_col and cosmetic_id:
                cursor.execute("""
                    INSERT INTO feedback (Users_ID, CosmeticID, Rating, CommentText, Date)
                    VALUES (%s, %s, %s, %s, CURRENT_DATE())
                """, (uid, cosmetic_id, rating, feedback_text))
            else:
                cursor.execute("""
                    INSERT INTO feedback (Users_ID, Rating, CommentText, Date)
                    VALUES (%s, %s, %s, CURRENT_DATE())
                """, (uid, rating, feedback_text))
            conn.commit()
            return jsonify({"status":"success","message":"Feedback ส่งสำเร็จ!"}), 200
        except mysql.connector.Error as db_err:
            if conn: conn.rollback()
            return jsonify({"status":"error","message": f"DB error: {db_err}"}), 500
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
    except Exception as e:
        return jsonify({"status": "error", "message": f"เกิดข้อผิดพลาดในการประมวลผล: {e}"}), 400

# -----------------------------
# MAKEUP LOOKS (compat)
# -----------------------------
def coerce_undertone(val: str):
    if not val: return None
    v = val.strip().lower()
    if v in {"warm tone","cool tone","neutral tone"}: return v.title()
    th = {"โทนอุ่น":"Warm Tone","โทนเย็น":"Cool Tone","โทนกลาง":"Neutral Tone"}
    return th.get(val, None)

UNDERTONE_TO_BRIGHTNESS = {
    "Warm Tone":   ["All","Medium","Deep"],
    "Cool Tone":   ["All","Fair","Medium"],
    "Neutral Tone":["All","Fair","Medium","Deep"]
}

@app.route('/ai/makeup_looks', methods=['GET'])
@jwt_required()
def get_makeup_looks():
    look_raw       = (request.args.get('category') or request.args.get('look') or request.args.get('lookCategory') or '').strip()
    q              = (request.args.get('q') or '').strip()
    undertone      = coerce_undertone(request.args.get('undertone') or request.args.get('skinTone'))
    with_cosmetics = (request.args.get('withCosmetics','false').lower() == 'true')
    limit_cos      = request.args.get('cosmeticsLimit', type=int) or 8

    canon = canon_look(look_raw)
    brightness_pref = UNDERTONE_TO_BRIGHTNESS.get(undertone, ["All","Fair","Medium","Deep"])

    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)

        where, params = [], []
        if canon:
            kws = LOOK_MAP[canon]["keywords"]
            where.append("(" + " OR ".join(["LOWER(lookName) LIKE %s"]*len(kws)) + ")")
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

        related = []
        if with_cosmetics:
            tones = [b.lower() for b in brightness_pref] + ['universal', 'all']
            placeholders = ",".join(["%s"]*len(tones))
            sql_cos = f"""
              SELECT c.CosmeticID, b.brandName, c.Name, c.Type,
                     COALESCE(c.ShadeCode, c.ShadeName) AS Shade,
                     c.Price, c.ImageURL, c.ProductLink,
                     c.suitableSkinTone
              FROM cosmetics c
              JOIN brand b ON b.brandID = c.BrandID
              WHERE LOWER(COALESCE(c.suitableSkinTone,'universal')) IN ({placeholders})
              ORDER BY c.Name ASC
              LIMIT %s
            """
            cursor.execute(sql_cos, tuple(tones+[limit_cos]))
            related = cursor.fetchall()
            for it in related:
                it['suitableSkinTone'] = canon_suitable_tone(it.get('suitableSkinTone'))

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
        return jsonify(payload), 200
    except Exception as e:
        print("makeup_looks error:", e)
        return jsonify({"error": "Failed to retrieve makeup looks"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# -----------------------------
# COSMETICS (list)
# -----------------------------
@app.route('/ai/cosmetics', methods=['GET'])
@jwt_required()
def get_all_cosmetics():
    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error":"Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                c.CosmeticID, c.Name, c.Type, c.Price,
                c.ImageURL, c.ProductLink, c.BrandID,
                b.brandName,
                c.suitableSkinTone, c.suitableLookType,
                c.Description, COALESCE(c.ShadeCode, c.ShadeName) AS Shade,
                c.Lab_L, c.Lab_a, c.Lab_b
            FROM cosmetics c
            JOIN brand b ON c.BrandID = b.brandID
            ORDER BY c.Name ASC
        """)
        rows = cursor.fetchall()
        for it in rows:
            it['suitableSkinTone'] = canon_suitable_tone(it.get('suitableSkinTone'))
        return jsonify(rows), 200
    except mysql.connector.Error as e:
        print(f"Error fetching cosmetics: {e}")
        return jsonify({"error":"Failed to retrieve cosmetics"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# -----------------------------
# Static palettes (serve from ./static)
# -----------------------------
@app.route('/palettes/<filename>')
def serve_palette_image(filename):
    return send_from_directory('static', filename)

@app.get("/ai/products/recommend")
@jwt_required()
def legacy_products_recommend():
    qs = request.query_string.decode()
    target = f"/ai/cosmetics/recommendations"
    if qs:
        target += f"?{qs}"
    return redirect(target, code=307)

# -----------------------------
# COSMETICS recommendations (ΔE00-aware; no retailer_offers, no best*)
# -----------------------------
@app.route('/ai/cosmetics/recommendations', methods=['GET'])
@jwt_required()
def get_recommended_cosmetics():
    skin_tone_q = (request.args.get('skinTone') or '').strip()
    budget_str  = (request.args.get('budget') or '').strip()
    look_raw    = (request.args.get('lookType') or request.args.get('look') or '').strip()
    style_id    = request.args.get('styleId', type=int)
    q           = (request.args.get('q') or '').strip()

    def parse_budget(s: str):
        if not s: return None, None
        s = s.replace(',', '').replace('บาท','').strip()
        if '+' in s:
            try: return int(s.split('+')[0].strip()), None
            except: return None, None
        if '-' in s:
            try:
                lo = int(s.split('-')[0].strip())
                hi = int(s.split('-')[1].strip())
                return lo, hi
            except: return None, None
        return None, None
    minp, maxp = parse_budget(budget_str)

    BUCKET_ANCHORS = {
        "Fair":   (80.0, 0.0, 20.0),
        "Medium": (60.0, 0.0, 20.0),
        "Brown":  (45.0, 0.0, 25.0),
        "Deep":   (30.0, 0.0, 20.0),
    }

    conn = cursor = None
    look_keywords = []
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error":"Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)

        # A) user skin
        uid = int(get_jwt_identity())
        cursor.execute("""
            SELECT SkinTone, Confidence, L_star, b_star
            FROM skintoneanalysis
            WHERE Users_ID=%s
            ORDER BY SkinToneAnalysisID DESC
            LIMIT 1
        """, (uid,))
        u_skin = cursor.fetchone() or {}

        user_brightness = (u_skin.get('SkinTone') or '').strip()
        if not user_brightness and skin_tone_q:
            user_brightness = skin_tone_q

        user_anchor = None
        user_L = u_skin.get('L_star'); user_b = u_skin.get('b_star')
        if user_L is not None and user_b is not None:
            try:
                user_anchor = (float(user_L), 0.0, float(user_b))
            except:
                user_anchor = None
        if user_anchor is None and user_brightness in BUCKET_ANCHORS:
            user_anchor = BUCKET_ANCHORS[user_brightness]

        # B) look keywords
        canon = canon_look(look_raw) if look_raw else None
        if canon:
            look_keywords = [k.lower() for k in LOOK_MAP[canon]["keywords"]]
        else:
            look_text = look_raw
            if not look_text and style_id:
                c2 = conn.cursor()
                c2.execute("SELECT lookName FROM makeuplook WHERE LookID=%s", (style_id,))
                r = c2.fetchone(); c2.close()
                if r: look_text = r[0] or ''
            if look_text:
                t = look_text.lower().strip()
                toks = [t] + [x for x in t.replace('/', ' ').replace(',', ' ').split() if len(x) >= 2]
                look_keywords = list(dict.fromkeys(toks))

        # C) WHERE
        where, params = [], []

        # tone filter: CSV-safe LIKE (",fair," in ",light, fair, medium,")
        if user_brightness:
            tone_clause, tone_param = tone_csv_like_clause("c.`suitableSkinTone`", user_brightness)
            where.append(f"({tone_clause} OR LOWER(COALESCE(c.`suitableSkinTone`,'universal')) IN (%s,%s))")
            params.extend([tone_param, 'universal', 'all'])

        if look_keywords:
            where.append("(" + " OR ".join(["LOWER(COALESCE(c.`suitableLookType`,'')) LIKE %s"]*len(look_keywords)) + ")")
            params.extend([f"%{kw}%" for kw in look_keywords])

        if q:
            where.append("(LOWER(c.`Name`) LIKE %s OR LOWER(b.`brandName`) LIKE %s OR LOWER(c.`Type`) LIKE %s)")
            params.extend([f"%{q.lower()}%", f"%{q.lower()}%", f"%{q.lower()}%"])

        if minp is not None:
            where.append("(c.Price >= %s)"); params.append(minp)
        if maxp is not None:
            where.append("(c.Price <= %s)"); params.append(maxp)

        # D) SELECT (no best* fields)
        sql = f"""
        SELECT
          c.CosmeticID, b.brandName, c.Name, c.Type,
          COALESCE(c.ShadeCode, c.ShadeName) AS Shade,
          c.Price, c.ImageURL, c.ProductLink,
          c.suitableSkinTone, c.suitableLookType,
          c.Lab_L, c.Lab_a, c.Lab_b,
          c.Description
        FROM cosmetics c
        JOIN brand b ON b.brandID = c.BrandID
        {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY c.Price ASC, c.Name ASC
        """
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        # E) rank
        skin_ai_conf = ((u_skin.get('Confidence') or 50) / 100.0)

        def source_trust_of(_row):
            # ไม่มีแหล่งร้านค้าแล้ว → ความเชื่อถือฐานกลาง ๆ
            return 0.7

        def conf_level(x: int) -> str:
            return "มั่นใจสูง" if x >= 75 else ("มั่นใจปานกลาง" if x >= 50 else "ควรทดลองเฉดใกล้เคียง")

        def product_family(t: str) -> str:
            if not t: return "other"
            v = (t or "").strip().lower()
            complexion = ["foundation","concealer","bb","cc","tinted","powder","cushion","contour","bronzer","base"]
            lips = ["lip","lipstick","gloss","oil","tint","stain","kit","liner"]
            blush = ["blush","cheek"]
            eyes = ["eyeshadow","eye shadow","eyeliner","mascara"]
            brows = ["brow","eyebrow"]
            if any(k in v for k in complexion): return "complexion"
            if any(k in v for k in lips): return "lips"
            if any(k in v for k in blush): return "blush"
            if any(k in v for k in eyes): return "eyes"
            if any(k in v for k in brows): return "brow"
            return "other"

        def color_bonus_for_family(family, deltaE):
            if deltaE is None:
                return 0.0, None
            if family == "complexion":
                if deltaE <= 2:  return 0.10, f"เฉดผิวแมตช์มาก (ΔE00={deltaE:.1f})"
                if deltaE <= 4:  return 0.07, f"เฉดใกล้ผิว (ΔE00={deltaE:.1f})"
                if deltaE <= 6:  return 0.04, f"เฉดพอใช้ (ΔE00={deltaE:.1f})"
                return 0.00, None
            if family in ("lips","blush"):
                if 15 <= deltaE <= 25: return 0.10, f"คอนทราสต์สวย (ΔE00={deltaE:.1f})"
                if 25 <  deltaE <= 40: return 0.07, f"คอนทราสต์เด่น (ΔE00={deltaE:.1f})"
                return 0.00, None
            if family == "eyes":
                if 20 <= deltaE <= 45: return 0.08, f"เฉดตาดูเด่น (ΔE00={deltaE:.1f})"
                if 45 <  deltaE <= 60: return 0.05, f"ลุคจัดชัด (ΔE00={deltaE:.1f})"
                return 0.00, None
            if family == "brow":
                if deltaE <= 5:   return 0.08, f"คิ้วกลืนผิว (ΔE00={deltaE:.1f})"
                if deltaE <= 12:  return 0.06, f"เฉดคิ้วใกล้เคียง (ΔE00={deltaE:.1f})"
                if deltaE <= 20:  return 0.03, f"เฉดคิ้วพอใช้ (ΔE00={deltaE:.1f})"
                return 0.00, None
            return 0.00, None

        augmented = []
        for r in rows:
            fam = product_family(r.get('Type') or r.get('Name') or "")
            reasons, badges = [], []
            rule = 0.0

            # tone match (CSV-safe + universal/all)
            tone_db = (r.get('suitableSkinTone') or '')
            if user_brightness:
                s_val = f",{user_brightness.lower()},"
                in_row = ("," + tone_db.lower().replace(" ", "") + ",")
                if (s_val.replace(" ", "") in in_row) or (tone_db.lower() in {"universal","all"}):
                    rule += 0.55
                    reasons.append("ตรงกับโทนความสว่างผิวของคุณ")

            if r.get('ImageURL'): rule += 0.05
            if r.get('Description'): rule += 0.05

            deltaE = None
            if (user_anchor is not None and
                r.get('Lab_L') is not None and r.get('Lab_a') is not None and r.get('Lab_b') is not None):
                try:
                    lab_prod = (float(r['Lab_L']), float(r['Lab_a']), float(r['Lab_b']))
                    deltaE = delta_e_ciede2000(user_anchor, lab_prod)
                except Exception:
                    deltaE = None

            color_bonus, color_reason = color_bonus_for_family(fam, deltaE)
            if color_bonus > 0:
                rule += color_bonus
                if color_reason:
                    reasons.append(color_reason)

            rule_conf = min(1.0, rule)
            s_trust = source_trust_of(r)

            hybrid = int(round(100 * (0.55 * rule_conf + 0.25 * skin_ai_conf + 0.20 * s_trust)))
            hybrid = max(0, min(100, hybrid))
            level = conf_level(hybrid)

            r['suitableSkinTone'] = canon_suitable_tone(r.get('suitableSkinTone'))
            item = {
                **r,
                "hybrid_confidence": hybrid,
                "confidence_level": level,
                "badges": list(dict.fromkeys(badges)),
                "reasons": reasons[:3]
            }
            if deltaE is not None:
                item["deltaE00"] = round(deltaE, 2)
            augmented.append(item)

        augmented.sort(key=lambda x: (
            -x['hybrid_confidence'],
            (float(x.get('Price')) if x.get('Price') is not None else 1e12),
            x.get('Name') or ""
        ))

        # --- Build palette payload (ตามโทนที่สรุปได้) ---
        rec_palettes = []
        pal_name = None
        if user_brightness and user_brightness in PALETTE_BY_TONE:
            pal_name = PALETTE_BY_TONE[user_brightness]

        # ถ้ายังไม่มี ลองดูจาก query ?skinTone=
        if not pal_name and skin_tone_q:
            st = (skin_tone_q or "").strip().title()  # fair -> Fair
            if st in PALETTE_BY_TONE:
                pal_name = PALETTE_BY_TONE[st]

        # แนบเฉพาะถ้าไฟล์อยู่จริงใน ./static
        if pal_name and os.path.exists(os.path.join("static", pal_name)):
            rec_palettes.append({
                "Tone": user_brightness or skin_tone_q,
                "ImageURL": pal_name
            })

        return jsonify({
            "recommendedCosmetics": augmented,
            "recommendedColorPalettes": rec_palettes
        }), 200

    except Exception as e:
        print("recommendations error:", e)
        return jsonify({"error":"Failed to retrieve recommendations"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# -----------------------------
# COSMETIC detail (no best offer)
# -----------------------------
@app.route('/ai/cosmetics/<int:cosmetic_id>', methods=['GET'])
@jwt_required()
def get_cosmetic_detail(cosmetic_id):
    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error":"Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT c.CosmeticID, b.brandName, c.Name, c.Type,
                   COALESCE(c.ShadeCode, c.ShadeName) AS Shade,
                   c.Price, c.ImageURL, c.ProductLink,
                   c.suitableSkinTone,
                   c.suitableLookType, c.Description,
                   c.Lab_L, c.Lab_a, c.Lab_b
            FROM cosmetics c
            JOIN brand b ON b.brandID = c.BrandID
            WHERE c.CosmeticID=%s
        """, (cosmetic_id,))
        item = cursor.fetchone()
        if not item:
            return jsonify({"error":"Cosmetic not found"}), 404
        item['suitableSkinTone'] = canon_suitable_tone(item.get('suitableSkinTone'))

        best_offer = None  # ไม่มี retailer_offers แล้ว
        return jsonify({"item": item, "bestOffer": best_offer}), 200
    except Exception as e:
        print("detail error:", e)
        return jsonify({"error":"Failed to retrieve cosmetic detail"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# -----------------------------
# Brands (list)
# -----------------------------
@app.route('/ai/brands', methods=['GET'])
@jwt_required()
def list_brands():
    conn = cursor = None
    try:
        conn = connect_db()
        if conn is None:
            return jsonify({"error":"Database connection failed"}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT brandID, brandName FROM brand ORDER BY brandName ASC")
        return jsonify(cursor.fetchall()), 200
    except Exception as e:
        print("brands error:", e)
        return jsonify({"error":"Failed to retrieve brands"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=False, threaded=True)

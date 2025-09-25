from flask import Flask, request, jsonify
import os
import mysql.connector as mysql
from werkzeug.utils import secure_filename
from datetime import datetime

UPLOAD_DIR = "static/images"  # โฟลเดอร์เสิร์ฟรูป
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB = dict(host="127.0.0.1", user="root", password="1234", database="db_miniprojectfinal")

app = Flask(__name__)

@app.post("/upload-image")
def upload_image():
    file = request.files.get("file")
    cosmetic_id = request.form.get("cosmetic_id")  # หรือ offer_id ถ้าจะผูกกับ retailer_offers
    if not file or not cosmetic_id:
        return jsonify({"error": "missing file or cosmetic_id"}), 400

    # ปลอดภัยชื่อไฟล์
    fname = secure_filename(file.filename)
    # ป้องกันชื่อซ้ำด้วย timestamp
    base, ext = os.path.splitext(fname)
    fname = f"{base}_{int(datetime.utcnow().timestamp())}{ext}"
    save_path = os.path.join(UPLOAD_DIR, fname)
    file.save(save_path)

    # URL ที่ฝั่งแอปจะเรียก (สมมติคุณเสิร์ฟ http://yourhost/static/images/…)
    public_url = f"/static/images/{fname}"

    conn = mysql.connect(**DB)
    cur = conn.cursor()
    try:
        cur.execute("UPDATE cosmetics SET ImageURL=%s WHERE CosmeticID=%s", (public_url, cosmetic_id))
        conn.commit()
    finally:
        cur.close(); conn.close()

    return jsonify({"ok": True, "image_url": public_url})

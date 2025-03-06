import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import os

# ✅ 1. โหลดโมเดล
MODEL_PATH = "resnet50_final_model_v2_edit_1.keras"

try:
    model = load_model(MODEL_PATH)
    model.build(input_shape=(None, 224, 224, 3))  # กำหนด Input Shape
    print("✅ โหลดโมเดลสำเร็จ!")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    exit()

# ✅ 2. ตรวจสอบเลเยอร์ของโมเดล
print("\n🔍 รายชื่อเลเยอร์ในโมเดล:")
for layer in model.layers:
    print(f"- {layer.name}: {layer}")

# ✅ 3. ดึง ResNet50 ออกมา และเพิ่ม GlobalAveragePooling2D
try:
    resnet_base = model.get_layer("resnet50")  # ดึง ResNet50 ออกมา
    feature_extractor = tf.keras.Sequential([
        resnet_base,
        GlobalAveragePooling2D()  # เพิ่ม Global Average Pooling เอง
    ])
    print("✅ ใช้ GlobalAveragePooling2D เป็น Feature Extractor")
except Exception as e:
    print(f"❌ ไม่สามารถกำหนด feature_extractor: {e}")
    exit()

# ✅ 4. ตรวจสอบว่า Feature Extractor ใช้งานได้หรือไม่
dummy_input = np.zeros((1, 224, 224, 3))
output = feature_extractor.predict(dummy_input, verbose=0)
print(f"✅ Test Feature Extractor Shape: {output.shape}")  # ควรได้ (1, 2048)

# ✅ 5. ตั้งค่าพาธที่เก็บรูปภาพ (โฟลเดอร์ Train + Augmented)
image_folder = "D:/AI-and-API-Celeb/Data_Image_celeb/DataTrain"
image_size = (224, 224)

feature_list = []
label_list = []

# ✅ 6. วนลูปทุกโฟลเดอร์ย่อย (แต่ละบุคคล)
for person_name in os.listdir(image_folder):
    person_folder = os.path.join(image_folder, person_name)

    if os.path.isdir(person_folder):  # ตรวจสอบว่าเป็นโฟลเดอร์
        print(f"📂 Processing: {person_name}")

        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".png")):  # รองรับ .JPG .PNG
                img_path = os.path.join(person_folder, filename)

                try:
                    # ✅ โหลดและแปลงภาพ
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                    img_array = np.expand_dims(img_array, axis=0)

                    # ✅ ใช้ feature_extractor ดึงเวกเตอร์ฟีเจอร์
                    feature = feature_extractor.predict(img_array, verbose=0)[0]
                    feature_list.append(feature)

                    # ✅ ใช้ชื่อโฟลเดอร์เป็น Label
                    label_list.append(person_name)

                except Exception as e:
                    print(f"❌ ไม่สามารถโหลดรูป {filename}: {e}")

# ✅ 7. แปลงเป็น NumPy Array
feature_database = np.array(feature_list)
label_database = np.array(label_list)

# ✅ 8. บันทึกเป็น `.npy`
np.save("feature_database.npy", feature_database)
np.save("label_database.npy", label_database)

# ✅ 9. ตรวจสอบขนาดของไฟล์ที่สร้าง
print(f"✅ สร้าง `feature_database.npy` สำเร็จ! ขนาด: {feature_database.shape}")  # ควรเป็น (N, 2048)
print(f"✅ สร้าง `label_database.npy` สำเร็จ! ขนาด: {label_database.shape}")  # ควรเป็น (N,)

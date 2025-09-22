import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import time

# --- Path และการตั้งค่าที่ต้องแก้ไข ---
# 1. Path ไปยังไฟล์โมเดล .pth ที่คุณเทรนเสร็จแล้ว
#    (ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกัน เช่น 'mobilenet_v2_skintonemodel.pth')
MODEL_PATH = 'D:/AI-and-API-Celeb/output_results/mobilenet_v2_skintonemodel.pth'

# 2. Path ไปยังรูปภาพที่คุณต้องการทดสอบ
TEST_IMAGE_PATH = 'D:/AI-and-API-Celeb/bcj4s2.jpg' # สมมติว่ามีรูปชื่อ test_image.jpg อยู่ในโฟลเดอร์นี้

# 3. ชื่อของโมเดลที่คุณใช้เทรน (ต้องตรงกับ SELECTED_MODEL ที่ใช้ใน test.py)
#    ตัวเลือก: 'resnet18', 'mobilenet_v2', 'efficientnet_b0', 'mobilenet_v3_small', 'shufflenet_v2_x0_5'
MODEL_ARCHITECTURE = 'mobilenet_v2' 

# 4. ชื่อคลาสของคุณ (ต้องเรียงลำดับเดียวกับตอนเทรน)
CLASS_NAMES = ['deep dark', 'fair', 'brown', 'medium'] # เรียงตามตัวอักษรของชื่อโฟลเดอร์ใน data_skintone

# --- Hyperparameters ที่ใช้ตอนเทรน (สำคัญสำหรับ Dropout layers) ---
# ต้องใช้ค่าเดียวกับตอนที่โมเดลถูกเทรน
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.4
DROPOUT_RATE_3 = 0.3

# กำหนด device เป็น CPU เสมอ เพราะโมเดลนี้ถูกเทรนบน CPU และคุณมีปัญหาเรื่อง GPU compatibility
device = torch.device("cpu")

# --- Model Loading Function (คัดลอกมาจาก test.py) ---
def get_model(model_name, num_classes):
    model = None
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False) # pretrained=False เพราะเราจะโหลด weight ของเราเอง
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1),
            nn.Linear(num_ftrs, 512),
            nn.Dropout(DROPOUT_RATE_2),
            nn.Linear(512, 128),
            nn.Dropout(DROPOUT_RATE_3),
            nn.Linear(128, num_classes)
        )
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False) # pretrained=False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1), 
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False) # pretrained=False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1), 
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=False) # pretrained=False
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1), 
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained=False) # pretrained=False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1), 
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported for loading. Check MODEL_ARCHITECTURE.")
    
    return model

# --- Data Transform สำหรับการ Inference ---
# ใช้ Transform เหมือนกับ 'val' ตอนเทรน
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Main Prediction Logic ---
if __name__ == '__main__':
    # ตรวจสอบว่าไฟล์โมเดลมีอยู่
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please check MODEL_PATH and ensure the model was saved successfully.")
    # ตรวจสอบว่าไฟล์รูปภาพมีอยู่
    elif not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image file not found at {TEST_IMAGE_PATH}")
        print("Please check TEST_IMAGE_PATH.")
    else:
        try:
            # 1. โหลดโมเดล
            print(f"Loading model: {MODEL_ARCHITECTURE} from {MODEL_PATH}")
            model = get_model(MODEL_ARCHITECTURE, len(CLASS_NAMES))
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # map_location=device สำคัญเมื่อโหลดบน CPU
            model.to(device)
            model.eval() # ตั้งค่าโมเดลเป็นโหมดประเมิน (สำคัญสำหรับ Dropout และ Batch Norm)
            print("Model loaded successfully.")

            # 2. โหลดและเตรียมรูปภาพ
            print(f"Processing image: {TEST_IMAGE_PATH}")
            image = Image.open(TEST_IMAGE_PATH).convert('RGB')
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # เพิ่มมิติ batch dimension
            input_batch = input_batch.to(device)

            # 3. ทำนายผล
            print("Making prediction...")
            start_time = time.time()
            with torch.no_grad(): # ไม่ต้องคำนวณ gradients ในโหมด inference
                output = model(input_batch)
            end_time = time.time()

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_index = torch.argmax(probabilities).item()
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index].item() * 100

            print(f"\n--- Prediction Results ---")
            print(f"Image: {os.path.basename(TEST_IMAGE_PATH)}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Prediction Time: {((end_time - start_time) * 1000):.2f} ms")

            print("\nAll Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {CLASS_NAMES[i]}: {prob.item()*100:.2f}%")

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            print("Please ensure your MODEL_ARCHITECTURE, CLASS_NAMES, and DROPOUT_RATEs match what was used during training.")
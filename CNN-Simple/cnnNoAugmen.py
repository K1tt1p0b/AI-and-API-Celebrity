import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ✅ ตั้งค่าพารามิเตอร์
img_height, img_width = 224, 224
batch_size = 32
epochs = 50
initial_learning_rate = 1e-4

# ✅ Normalize ข้อมูล (ไม่มี Data Augmentation)
train_datagen = ImageDataGenerator(rescale=1./255)

# ✅ แบ่ง `DataTest` ออกเป็น 2 ส่วน (50% Validation, 50% Test)
validation_test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

# ✅ โหลดข้อมูล Training (80% ของข้อมูลทั้งหมด)
train_generator = train_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTrain',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ ใช้ 50% ของ DataTest เป็น Validation Set (10% ของข้อมูลทั้งหมด)
validation_generator = validation_test_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTest',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # ✅ ใช้ 50% ของ DataTest เป็น Validation
    shuffle=False
)

# ✅ ใช้ 50% ของ DataTest เป็น Test Set (10% ของข้อมูลทั้งหมด)
test_generator = validation_test_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTest',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # ✅ ใช้ 50% ของ DataTest เป็น Test
    shuffle=False
)

# ✅ สร้าง CNN Model แบบ Simple (โครงสร้างคล้าย MobileNetV2)
def build_simple_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2 (Depthwise Separable Conv)
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Feature Extraction
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # ลด Overfitting

        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ✅ สร้างโมเดล
model = build_simple_cnn((img_height, img_width, 3), train_generator.num_classes)

# ✅ Compile Model
optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_model_simple_cnn.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ✅ Train Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,  # ✅ ใช้ 50% ของ DataTest เป็น Validation
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# ✅ Save Final Model
model.save('cnn_final_simple_model.keras')
print("Final model saved!")

# ✅ Evaluate Results (ใช้ 50% ของ DataTest เป็น Test Set)
y_true = []
y_pred = []
for i in range(len(test_generator)):
    x_test, y_test = test_generator[i]
    predictions = model.predict(x_test)
    y_true.extend(np.argmax(y_test, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print evaluation metrics
print("\n📊 *Evaluation Metrics on Test Data:*")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")

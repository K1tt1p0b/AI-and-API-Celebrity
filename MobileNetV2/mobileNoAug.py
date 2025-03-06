import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# ✅ ตั้งค่าพารามิเตอร์
img_height, img_width = 224, 224
batch_size = 32
epochs = 100
initial_learning_rate = 1e-4  # ปรับให้สมดุล

# ✅ ไม่ใช้ Data Augmentation (แค่ Normalize)
train_datagen = ImageDataGenerator(rescale=1./255)

# ✅ ใช้ DataTest แต่แบ่งออกเป็น 2 ส่วน (50% Validation, 50% Test)
validation_test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

# ✅ โหลดข้อมูล (DataTrain ใช้สำหรับ Training)
train_generator = train_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTrain',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ ใช้ 50% ของ DataTest เป็น Validation Set
validation_generator = validation_test_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTest',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # ✅ ใช้ 50% ของ DataTest เป็น Validation
    shuffle=False
)

# ✅ ใช้ 50% ของ DataTest เป็น Test Set
test_generator = validation_test_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTest',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # ✅ ใช้ 50% ของ DataTest เป็น Test
    shuffle=False
)

# ✅ ใช้ MobileNetV2 (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model.layers[:-100]:  
    layer.trainable = False  # ล็อก Layer แรก ลด Overfitting

# ✅ Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  
    layers.Dropout(0.5),  
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# ✅ Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr * 1.1  
    elif epoch < 50:
        return lr * 0.9  
    else:
        return lr * 0.8  

lr_callback = LearningRateScheduler(lr_scheduler)

# ✅ Compile Model
optimizer = AdamW(learning_rate=initial_learning_rate, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_mobilenetV2.keras', monitor='val_loss', save_best_only=True, verbose=1)

# ✅ Train Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,  # ✅ ใช้ 50% ของ DataTest เป็น Validation
    callbacks=[early_stopping, reduce_lr, model_checkpoint, lr_callback]
)

# ✅ Save Final Model
model.save('mobilenet_final_model_v2.keras')
print("Final model saved!")

# ✅ Evaluate Results (ใช้ 50% ของ DataTest เป็น Test Set)
y_true = []
y_pred = []
for i in range(len(test_generator)):
    x_test, y_test = test_generator[i]
    predictions = model.predict(x_test)
    y_true.extend(np.argmax(y_test, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n📊 *Evaluation Metrics on Test Data:*")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")

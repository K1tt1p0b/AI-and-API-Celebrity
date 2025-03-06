import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ✅ ตั้งค่าพารามิเตอร์
img_height, img_width = 224, 224
batch_size = 32
epochs = 50
initial_learning_rate = 1e-4

# ✅ Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  # Use 20% data for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize test images

# ✅ Load Data
train_generator = train_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTrain',  # Path to training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

validation_generator = test_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTest',  # Path to test/validation data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Don't shuffle validation data
)

# ✅ สร้าง CNN Model แบบ Simple
model = models.Sequential([  
    # ชั้น Convolutional Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # ชั้น Convolutional Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer for multi-class classification
])

# ✅ Compile Model
optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
    metrics=['accuracy']
)

# ✅ Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop early if no improvement in 10 epochs
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_model_simple_cnn.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1  # Print message when saving best model
)

# ✅ Train Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# ✅ Save Final Model
model.save('cnn_final_simple_model.keras')
print("Final model saved!")

# ✅ Evaluate Results
test_generator = test_datagen.flow_from_directory(
    'F:/CelebAI100/Data_Image_celeb/DataTest',  # Path to test data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Don't shuffle test data
)

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

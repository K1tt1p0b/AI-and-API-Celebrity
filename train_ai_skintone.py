import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn # ยังคง import ไว้ได้ แต่จะไม่มีผลเมื่อ device เป็น CPU
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ตั้งค่า CUDA_LAUNCH_BLOCKING เพื่อช่วยในการ Debug ถ้ายังเจอ CUDA Error (ไม่จำเป็นเมื่อใช้ CPU)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# cudnn.benchmark = True # ไม่จำเป็นสำหรับ CPU

plt.ion()

data_dir = 'D:/AI-and-API-Celeb/data_skintone'
class_mapping = {
    'light': 'fair',
    'dark': 'deep dark',
    'mid-light': 'medium',
    'mid-dark': 'brown'
}

output_results_dir = 'D:/AI-and-API-Celeb/output_results'
os.makedirs(output_results_dir, exist_ok=True)

# --- Hyperparameters (ปรับได้ตามต้องการ) ---
NUM_EPOCHS = 25
BATCH_SIZE = 32 # สามารถใช้ค่านี้ได้ แต่ถ้า RAM ไม่พออาจจะต้องลดลงอีกสำหรับ CPU
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.4
DROPOUT_RATE_3 = 0.3 # อาจไม่ถูกใช้ครบทุก Dropout สำหรับโมเดลที่เล็กลง

# เลือกโมเดลที่คุณต้องการใช้จากตรงนี้
# ตั้งค่าเป็น 'mobilenet_v2' โดยตรง
SELECTED_MODEL = 'mobilenet_v2' 

# --- Data Transforms ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Custom Image Folder Class ---
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        
        original_class_to_idx = self.class_to_idx.copy()

        self.class_to_idx = {class_mapping[k]: v for k, v in original_class_to_idx.items() if k in class_mapping}
        self.classes = [class_mapping[c] for c in self.classes if c in class_mapping]

        if len(self.classes) != len(class_mapping):
            print("Warning: Mapped classes do not match the expected number. Check class_mapping and folder names in your dataset.")
            print(f"Original classes found by ImageFolder: {original_class_to_idx.keys()}")
            print(f"Mapped classes after filtering: {self.classes}")

# --- Model Loading Function ---
def get_model(model_name, num_classes):
    """
    Loads a pretrained model and modifies its final classification layer.
    """
    model = None
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
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
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            # สามารถเพิ่ม Dropout ได้ที่นี่ หรือจะใช้แค่ Linear ก็ได้ถ้า MobileNet มี Dropout ในตัวอยู่แล้ว
            nn.Dropout(DROPOUT_RATE_1), # เพิ่ม Dropout เพิ่มเติม (ปรับค่าได้)
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            # EfficientNet มี Dropout ในตัวอยู่แล้วในส่วน classifier[0] (เป็น nn.Dropout)
            # สามารถปรับ DROPOUT_RATE_1 เพื่อใช้กับ Dropout layer ของ EfficientNet เองได้
            nn.Dropout(DROPOUT_RATE_1), # เพิ่ม Dropout เพิ่มเติม (ปรับค่าได้)
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[3].in_features # สำหรับ MobileNetV3-Small
        model.classifier[3] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1), # สามารถปรับหรือใช้แค่ Linear ก็ได้
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_1), # เพิ่ม Dropout เพิ่มเติม
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose from 'resnet18', 'mobilenet_v2', 'efficientnet_b0', 'mobilenet_v3_small', 'shufflenet_v2_x0_5'.")
    
    return model

# --- Plotting Functions ---
def plot_history(train_values, val_values, title, ylabel, filename, output_dir):
    epochs = range(len(train_values))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, label='Training')
    plt.plot(epochs, val_values, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_training_history(train_loss, val_loss, train_acc, val_acc, output_dir):
    train_loss_cpu = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in train_loss]
    val_loss_cpu = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in val_loss]
    train_acc_cpu = [acc.item() if isinstance(acc, torch.Tensor) else acc for acc in train_acc]
    val_acc_cpu = [acc.item() if isinstance(acc, torch.Tensor) else acc for acc in val_acc]

    plot_history(train_loss_cpu, val_loss_cpu, 'Training and Validation Loss', 'Loss', 'loss_plot.png', output_dir)
    plot_history(train_acc_cpu, val_acc_cpu, 'Training and Validation Accuracy', 'Accuracy', 'accuracy_plot.png', output_dir)

# --- Training Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs, output_dir, dataloaders, dataset_sizes, device):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_loss = float('inf')

        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_loss = epoch_loss
                        torch.save(model.state_dict(), best_model_params_path)
            
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        print(f'Best val Loss: {best_loss:.4f}')
        
        model.load_state_dict(torch.load(best_model_params_path))
        plot_training_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history, output_dir)

    return model

# --- Evaluation Functions ---
def visualize_model(model, test_folder, data_transforms, device, class_names):
    was_training = model.training
    model.eval()
    label = []
    prediction = []

    for path in os.listdir(test_folder):
        level = os.path.join(test_folder, path)
        for img_name in os.listdir(level):
            label.append(path)
            img_path = os.path.join(level, img_name)
            img = Image.open(img_path).convert('RGB')
            img = data_transforms['val'](img)
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                outputs = model(img)
                _, preds = torch.max(outputs, 1)
                prediction.append(class_names[preds[0]])

    count = sum(label[i] == prediction[i] for i in range(len(label)))
    accuracy = count / len(label)
    print(f'Accuracy on test set: {accuracy}')
    model.train(mode=was_training)

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report on Validation Set:")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(len(class_names) + 2, len(class_names) + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_results_dir, 'confusion_matrix.png'))
    plt.show()

    return report

# --- Main execution block ---
if __name__ == '__main__':
    # กำหนด device เป็น CPU เสมอ
    device = torch.device("cpu") 
    print(f"Device: {device} (Forced to CPU)")

    # เมื่อใช้ CPU ให้กำหนด num_workers เป็น 0 เพื่อหลีกเลี่ยงปัญหา multiprocessing บน Windows
    num_workers_for_dataloaders = 0 

    all_data = CustomImageFolder(data_dir, data_transforms['train'])

    total_size = len(all_data)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_data, val_data = torch.utils.data.random_split(all_data, [train_size, val_size])

    val_data.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_for_dataloaders),
        'val': torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers_for_dataloaders),
    }

    dataset_sizes = {
        'train': len(train_data),
        'val': len(val_data),
    }

    class_names = all_data.classes

    print(f"Class Names: {class_names}")
    print(f"Dataset Sizes: {dataset_sizes}")
    print("-" * 30)

    # โหลดโมเดลตามที่ SELECTED_MODEL ถูกตั้งค่าไว้ (ตอนนี้คือ mobilenet_v2)
    model_ft = get_model(SELECTED_MODEL, len(class_names))

    # ย้ายโมเดลไปยัง CPU
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print(f"\nStarting Model Training with {SELECTED_MODEL} (on CPU)...")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 
                            NUM_EPOCHS, output_results_dir,
                           dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)

    print(f"\nEvaluating Model on Validation Set (with {SELECTED_MODEL})...")
    evaluation_report = evaluate_model(model_ft, dataloaders['val'], device, class_names)

    model_file_path = os.path.join(output_results_dir, f'{SELECTED_MODEL}_skintonemodel.pth')
    torch.save(model_ft.state_dict(), model_file_path)
    print(f'\nModel saved to {model_file_path}')
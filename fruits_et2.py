# Importuri necesare
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, ResNet18_Weights


# ================== ÎNCĂRCAREA DATELOR ==================
# Funcția de încărcare a datelor este predefinită și nu va fi schimbată
def load_images_from_folder(folder):
    import os
    from PIL import Image
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(subdir, file)
                label = os.path.basename(subdir)
                img = Image.open(path).resize((32, 32))  # Redimensionare
                images.append(np.array(img).flatten())  # Aplatizare
                labels.append(label)
    return np.array(images), np.array(labels)

# Încărcarea imaginilor și etichetelor
training_images, train_labels = load_images_from_folder('fruits-360/Training')
test_images, test_labels = load_images_from_folder('fruits-360/Test')

print(f"Number of training images: {len(training_images)}")
print(f"Number of test images: {len(test_images)}")


unique_labels = sorted(set(train_labels))  # Listează toate etichetele unice
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Transformarea etichetelor în valori numerice
train_labels_numeric = np.array([label_to_index[label] for label in train_labels])
test_labels_numeric = np.array([label_to_index[label] for label in test_labels])

# ================== SELECȚIE DE ATRIBUTE ==================
# Filtrare până la 128 de atribute
selector = SelectKBest(f_classif, k=128)
X_train_selected = selector.fit_transform(training_images, train_labels_numeric)
X_test_selected = selector.transform(test_images)

# Standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Conversie la tensori PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels_numeric, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels_numeric, dtype=torch.long)

print("Numarul de atribute selectate: ", X_train_selected.shape[1])

# ================== DEFINIREA ARHITECTURII MLP ==================
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Dimensiunea intrării și numărul de clase
input_size = X_train_tensor.shape[1]  # Numărul de atribute selectate (128)
num_classes = len(set(train_labels))  # 80 de clase în Fruits-360

# Inițializarea modelului
model = MLP(input_size, num_classes)

# ================== CONFIGURAREA ANTRENĂRII ==================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Parametrii pentru antrenare
num_epochs = 50
batch_size = 64

# Crearea DataLoader-elor pentru antrenare și testare
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ================== ANTRENAREA MODELULUI ==================
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    # ================== GRAFICELE PENTRU LOSS ȘI ACURATEȚE ==================
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))

# Loss
plt.plot(epochs, train_losses, label='Train Loss', linestyle='--', marker='o')
plt.plot(epochs, test_losses, label='Validation Loss', linestyle='--', marker='x')

# Accurac
plt.plot(epochs, train_accuracies, label='Train Accuracy', linestyle='-', marker='s')
plt.plot(epochs, test_accuracies, label='Validation Accuracy', linestyle='-', marker='d')

# Adăugarea titlurilor și legendelor
plt.title('Loss and Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Salvarea și afișarea graficului
plt.savefig('fruits_etapa2/loss_accuracy_curves.png')
plt.show()

# ================== RAPORT FINAL ==================
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

print("Classification Report:\n", classification_report(y_true, y_pred))

# Matricea de confuzie
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fruits_etapa2/confusion_matrix.png')
plt.show()

# Afisarea a 10 exemple de clasificare corectă și 10 exemple de clasificare greșită
correct_images = []
correct_labels = []
incorrect_images = []
incorrect_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                correct_images.append(inputs[i].numpy())
                correct_labels.append(predicted[i].item())
            else:
                incorrect_images.append(inputs[i].numpy())
                incorrect_labels.append(predicted[i].item())

    correct_images = np.array(correct_images)
    correct_labels = np.array(correct_labels)
    incorrect_images = np.array(incorrect_images)
    incorrect_labels = np.array(incorrect_labels)

    print("Numărul de exemple corecte:", len(correct_images))
    print("Numărul de exemple greșite:", len(incorrect_images))



# ================== CERINTA 3.2 ==================

# ================== LINIARIZAREA IMAGINILOR ==================
X_train_images = training_images / 255.0  # Normalizare
X_test_images = test_images / 255.0  # Normalizare

# Conversie la tensori PyTorch
X_train_tensor_images = torch.tensor(X_train_images, dtype=torch.float32)
y_train_tensor_images = torch.tensor(train_labels_numeric, dtype=torch.long)
X_test_tensor_images = torch.tensor(X_test_images, dtype=torch.float32)
y_test_tensor_images = torch.tensor(test_labels_numeric, dtype=torch.long)

# ================== DEFINIREA ARHITECTURII MLP DIRECT PE IMAGINI ==================
class MLPImages(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPImages, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Inițializarea modelului
input_size_images = X_train_tensor_images.shape[1]  # Dimensiunea imaginii liniarizate
model_images = MLPImages(input_size_images, num_classes)

# ================== CONFIGURAREA ANTRENĂRII ==================
criterion_images = nn.CrossEntropyLoss()
optimizer_images = optim.Adam(model_images.parameters(), lr=0.001)

# Parametrii pentru antrenare
num_epochs = 50
batch_size = 64

# Crearea DataLoader-elor pentru antrenare și testare
train_dataset_images = torch.utils.data.TensorDataset(X_train_tensor_images, y_train_tensor_images)
test_dataset_images = torch.utils.data.TensorDataset(X_test_tensor_images, y_test_tensor_images)

train_loader_images = torch.utils.data.DataLoader(train_dataset_images, batch_size=batch_size, shuffle=True)
test_loader_images = torch.utils.data.DataLoader(test_dataset_images, batch_size=batch_size, shuffle=False)

# ================== ANTRENAREA MODELULUI PE IMAGINI ==================
train_losses_images = []
test_losses_images = []
train_accuracies_images = []
test_accuracies_images = []

for epoch in range(num_epochs):
    model_images.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader_images:
        optimizer_images.zero_grad()
        outputs = model_images(inputs)
        loss = criterion_images(outputs, labels)
        loss.backward()
        optimizer_images.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader_images)
    train_accuracy = 100.0 * correct / total
    train_losses_images.append(train_loss)
    train_accuracies_images.append(train_accuracy)

    model_images.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader_images:
            outputs = model_images(inputs)
            loss = criterion_images(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader_images)
    test_accuracy = 100.0 * correct / total
    test_losses_images.append(test_loss)
    test_accuracies_images.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

# ================== GRAFICELE PENTRU MLP PE IMAGINI ==================
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))

# Loss
plt.plot(epochs, train_losses_images, label='Train Loss (Images)', linestyle='--', marker='o')
plt.plot(epochs, test_losses_images, label='Validation Loss (Images)', linestyle='--', marker='x')

# Accuracy
plt.plot(epochs, train_accuracies_images, label='Train Accuracy (Images)', linestyle='-', marker='s')
plt.plot(epochs, test_accuracies_images, label='Validation Accuracy (Images)', linestyle='-', marker='d')

# Adăugarea titlurilor și legendelor
plt.title('Loss and Accuracy Curves for MLP on Images')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Salvarea și afișarea graficului
plt.savefig('fruits_etapa2/loss_accuracy_curves_images.png')
plt.show()

# ================== RAPORT FINAL PENTRU MLP PE IMAGINI ==================

model_images.eval()
y_pred_images = []
y_true_images = []

with torch.no_grad():
    for inputs, labels in test_loader_images:
        outputs = model_images(inputs)
        _, predicted = outputs.max(1)
        y_pred_images.extend(predicted.tolist())
        y_true_images.extend(labels.tolist())

print("Classification Report (Images):\n", classification_report(y_true_images, y_pred_images))

# Matricea de confuzie
cm_images = confusion_matrix(y_true_images, y_pred_images)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_images, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix (Images)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fruits_etapa2/confusion_matrix_images_mlp.png')
plt.show()

# ===================== CERINTA 3.3 =====================
# ===================== CNN =====================

# Crearea unei clase personalizate pentru încărcarea datelor
class FruitsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.reshape(-1, 3, 32, 32) / 255.0  # Reshape pentru RGB
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(torch.tensor(image, dtype=torch.float32))

        return image, label

# Transformări fără augmentare
transform_no_aug = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizare
])

# Transformări cu augmentare
transform_with_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Crearea seturilor de date
train_dataset_no_aug = FruitsDataset(training_images, train_labels_numeric, transform=transform_no_aug)
test_dataset_no_aug = FruitsDataset(test_images, test_labels_numeric, transform=transform_no_aug)

train_dataset_with_aug = FruitsDataset(training_images, train_labels_numeric, transform=transform_with_aug)

# DataLoader
batch_size = 64
train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=batch_size, shuffle=True)
test_loader_no_aug = DataLoader(test_dataset_no_aug, batch_size=batch_size, shuffle=False)
train_loader_with_aug = DataLoader(train_dataset_with_aug, batch_size=batch_size, shuffle=True)

# ===================== DEFINIREA CNN =====================

class DeepConvNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Inițializarea modelului
num_classes = len(unique_labels)
model_cnn = DeepConvNet(num_classes)

# ===================== CONFIGURAREA ANTRENĂRII =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)
num_epochs = 20

# ===================== FUNCȚIA DE ANTRENARE ȘI VALIDARE =====================

def train_and_evaluate(model, train_loader, test_loader, num_epochs):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    return train_losses, test_losses, train_accuracies, test_accuracies

# ===================== ANTRENAREA ȘI EVALUAREA =====================
# Fără augmentări
train_losses_no_aug, test_losses_no_aug, train_acc_no_aug, test_acc_no_aug = train_and_evaluate(
    model_cnn, train_loader_no_aug, test_loader_no_aug, num_epochs)

# Cu augmentări
model_cnn_aug = DeepConvNet(num_classes)
optimizer_aug = optim.Adam(model_cnn_aug.parameters(), lr=0.001)
train_losses_aug, test_losses_aug, train_acc_aug, test_acc_aug = train_and_evaluate(
    model_cnn_aug, train_loader_with_aug, test_loader_no_aug, num_epochs)

# ===================== GRAFICE =====================

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))
plt.plot(epochs, train_losses_no_aug, label='Train Loss (No Aug)', linestyle='--', marker='o')
plt.plot(epochs, test_losses_no_aug, label='Test Loss (No Aug)', linestyle='--', marker='x')
plt.plot(epochs, train_acc_no_aug, label='Train Accuracy (No Aug)', linestyle='-', marker='o')
plt.plot(epochs, test_acc_no_aug, label='Test Accuracy (No Aug)', linestyle='-', marker='x')
plt.title('Curves for CNN no Aug')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('fruits_etapa2/cnn_no_curves.png')
plt.show()

plt.figure(figsize=(12, 8))

plt.plot(epochs, train_acc_aug, label='Train Accuracy (With Aug)', linestyle='-', marker='s')
plt.plot(epochs, test_acc_aug, label='Test Accuracy (With Aug)', linestyle='-', marker='d')
plt.plot(epochs, train_losses_aug, label='Train Loss (With Aug)', linestyle='--', marker='s')
plt.plot(epochs, test_losses_aug, label='Test Loss (With Aug)', linestyle='--', marker='d')
plt.title('Curves for CNN (With Aug)')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('fruits_etapa2/cnn_aug_curves.png')
plt.show()



# ===================== RAPORT FINAL PENTRU CNN =====================

model_images.eval()
y_pred_images = []
y_true_images = []

with torch.no_grad():
    for inputs, labels in test_loader_no_aug:
        outputs = model_cnn(inputs)
        _, predicted = outputs.max(1)
        y_pred_images.extend(predicted.tolist())
        y_true_images.extend(labels.tolist())

print("Classification Report (CNN - No Augmentation):\n", classification_report(y_true_images, y_pred_images))

# Matricea de confuzie
cm_images = confusion_matrix(y_true_images, y_pred_images)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_images, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix (CNN - No Augmentation)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fruits_etapa2/confusion_matrix_cnn_no_aug.png')
plt.show()

# ===================== RAPORT FINAL PENTRU CNN CU AUGMENTĂRI =====================

model_images.eval()
y_pred_images = []
y_true_images = []

with torch.no_grad():
    for inputs, labels in test_loader_no_aug:
        outputs = model_cnn_aug(inputs)
        _, predicted = outputs.max(1)
        y_pred_images.extend(predicted.tolist())
        y_true_images.extend(labels.tolist())

print("Classification Report (CNN - With Augmentation):\n", classification_report(y_true_images, y_pred_images, zero_division=1))

# Matricea de confuzie
cm_images = confusion_matrix(y_true_images, y_pred_images)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_images, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix (CNN - With Augmentation)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fruits_etapa2/confusion_matrix_cnn_aug.png')


# ===================== CERINTA 3.4 =====================

transform_resnet = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensionare la 32x32
    transforms.ToTensor(),       # Convertire în tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizare
])

# Dataset personalizat pentru ResNet
class FruitsDatasetResNet(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.reshape(-1, 32, 32, 3) / 255.0  # Asigură-te că imaginile sunt RGB
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)  # Convertire la float
        label = self.labels[idx]

        if self.transform:
            # Transformările torchvision cer imagini PIL sau tensors
            image = self.transform(Image.fromarray((image * 255).astype(np.uint8)))

        return image, label



# Crearea dataset-urilor pentru ResNet
train_dataset_resnet = FruitsDatasetResNet(training_images, train_labels_numeric, transform=transform_resnet)
test_dataset_resnet = FruitsDatasetResNet(test_images, test_labels_numeric, transform=transform_resnet)

# DataLoader
batch_size = 64
train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=batch_size, shuffle=True)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=batch_size, shuffle=False)

# ================== ÎNCĂRCAREA MODELULUI RESNET-18 ==================
# Încărcare model pre-antrenat
#model_resnet = resnet18(pretrained=True)
model_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Înghețarea straturilor
for param in model_resnet.parameters():
    param.requires_grad = True

# Înlocuirea stratului Fully Connected
num_features = model_resnet.fc.in_features
num_classes = len(unique_labels)
model_resnet.fc = nn.Linear(num_features, num_classes)

# Optimizator și scheduler
optimizer_resnet = optim.SGD(model_resnet.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer_resnet, step_size=5, gamma=0.1)

# Funcția de pierdere
criterion_resnet = nn.CrossEntropyLoss()

# ================== ANTRENAREA CU STRATURI ÎNGHEȚATE ==================
num_epochs_frozen = 10
train_losses_resnet = []
test_losses_resnet = []
train_accuracies_resnet = []
test_accuracies_resnet = []

for epoch in range(num_epochs_frozen):
    model_resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader_resnet:
        optimizer_resnet.zero_grad()
        outputs = model_resnet(inputs)
        loss = criterion_resnet(outputs, labels)
        loss.backward()
        optimizer_resnet.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader_resnet)
    train_accuracy = 100.0 * correct / total
    train_losses_resnet.append(train_loss)
    train_accuracies_resnet.append(train_accuracy)

    # Evaluare
    model_resnet.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader_resnet:
            outputs = model_resnet(inputs)
            loss = criterion_resnet(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader_resnet)
    test_accuracy = 100.0 * correct / total
    test_losses_resnet.append(test_loss)
    test_accuracies_resnet.append(test_accuracy)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs_frozen}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

# ================== DEBLOCAREA STRATURILOR ȘI FINE-TUNING ==================
for param in model_resnet.parameters():
    param.requires_grad = True

# Optimizator cu rată mai mică
optimizer_resnet = optim.SGD(model_resnet.parameters(), lr=0.001, momentum=0.9)

num_epochs_unfrozen = 10
for epoch in range(num_epochs_unfrozen):
    model_resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader_resnet:
        optimizer_resnet.zero_grad()
        outputs = model_resnet(inputs)
        loss = criterion_resnet(outputs, labels)
        loss.backward()
        optimizer_resnet.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader_resnet)
    train_accuracy = 100.0 * correct / total
    train_losses_resnet.append(train_loss)
    train_accuracies_resnet.append(train_accuracy)

    # Evaluare
    model_resnet.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader_resnet:
            outputs = model_resnet(inputs)
            loss = criterion_resnet(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader_resnet)
    test_accuracy = 100.0 * correct / total
    test_losses_resnet.append(test_loss)
    test_accuracies_resnet.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs_unfrozen}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

# Afisarea rapoartelor de performanță
model_resnet.eval()
y_pred_resnet = []
y_true_resnet = []

with torch.no_grad():
    for inputs, labels in test_loader_resnet:
        outputs = model_resnet(inputs)
        _, predicted = outputs.max(1)
        y_pred_resnet.extend(predicted.tolist())
        y_true_resnet.extend(labels.tolist())

print("Classification Report (ResNet):\n", classification_report(y_true_resnet, y_pred_resnet))

# Matricea de confuzie
cm_resnet = confusion_matrix(y_true_resnet, y_pred_resnet)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix (ResNet)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fruits_etapa2/confusion_matrix_resnet.png')
plt.show()


# ================== GRAFICE ==================
epochs_total = range(1, num_epochs_frozen + num_epochs_unfrozen + 1)

plt.figure(figsize=(12, 8))
plt.plot(epochs_total, train_losses_resnet, label='Train Loss (ResNet)', linestyle='--', marker='o')
plt.plot(epochs_total, test_losses_resnet, label='Test Loss (ResNet)', linestyle='--', marker='x')
plt.title('Loss Curves for ResNet Fine-tuning')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('resnet_finetuning_loss.png')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(epochs_total, train_accuracies_resnet, label='Train Accuracy (ResNet)', linestyle='-', marker='s')
plt.plot(epochs_total, test_accuracies_resnet, label='Test Accuracy (ResNet)', linestyle='-', marker='d')
plt.title('Accuracy Curves for ResNet Fine-tuning')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('resnet_finetuning_accuracy.png')
plt.show()




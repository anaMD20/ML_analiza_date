import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import sys
sys.path.append('fashion-mnist/utils')
import numpy as np
from mnist_reader import load_mnist
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# Citirea datelor
train_images, train_labels = load_mnist('fashion-mnist/data/fashion', kind='train')
test_images, test_labels = load_mnist('fashion-mnist/data/fashion', kind='t10k')

# Reshape și normalizare
train_images = train_images.reshape((train_images.shape[0], -1)) / 255.0
test_images = test_images.reshape((test_images.shape[0], -1)) / 255.0

# Reshape și normalizare pentru CNN
train_images_cnn = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images_cnn = test_images.reshape(-1, 28, 28, 1) / 255.0

# # Selectarea a maxim 64 de atribute
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=64)  # Selectăm exact 64 de atribute
X_train = selector.fit_transform(train_images, train_labels)
X_test = selector.transform(test_images)


# Afișarea numărului de atribute extrase
print(f"Numărul de atribute extrase: {X_train.shape[1]}")

# Standardizarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definirea arhitecturii MLP
mlp_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compilarea modelului
mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Antrenarea modelului
history = mlp_model.fit(
    X_train_scaled, train_labels,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    verbose=2
)

# Evaluarea modelului
loss, accuracy = mlp_model.evaluate(X_test_scaled, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Afișarea rapoartelor de performanță
predictions = mlp_model.predict(X_test_scaled)
y_pred = predictions.argmax(axis=1)
print("Classification Report:\n", classification_report(test_labels, y_pred))

# Matricea de confuzie
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for MLP')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fashion_mnist_images/confusion_matrix.png')

# Graficul combinat al erorii și acurateței
plt.figure(figsize=(10, 6))

# Curbele de eroare
plt.plot(history.history['loss'], label='Train Loss', linestyle='--', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', marker='x')

# Curbele de acuratețe
plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='-', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='x')

plt.title('Loss and Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
# plt.show()

plt.savefig('fashion_mnist_images/curbe.png')

# Salvarea modelului
mlp_model.save('fashion_mnist_images/mlp_model.h5')
print("Modelul a fost salvat cu succes!")

# Afișarea a 10 exemple de predicții corecte și 10 exemple de predicții greșite
correct = np.nonzero(y_pred == test_labels)[0]
incorrect = np.nonzero(y_pred != test_labels)[0]

plt.figure(figsize=(10, 6))
for i, correct in enumerate(correct[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f"Predicted: {y_pred[correct]}, Actual: {test_labels[correct]}")
    plt.xticks([])
    plt.yticks([])
plt.suptitle('Corecte')
plt.savefig('fashion_mnist_images/corecte.png')

plt.figure(figsize=(10, 6))
for i, incorrect in enumerate(incorrect[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f"Predicted: {y_pred[incorrect]}, Actual: {test_labels[incorrect]}")
    plt.xticks([])
    plt.yticks([])

plt.suptitle('Greșite')
plt.savefig('fashion_mnist_images/gresite.png')

# ================= CERINȚA 3.2 ==================
# Standardizarea datelor pentru imagini liniare
scaler_images = StandardScaler()
train_images_scaled = scaler_images.fit_transform(train_images)
test_images_scaled = scaler_images.transform(test_images)

# Definirea arhitecturii MLP pentru imagini liniare
mlp_model_images = Sequential([
    Dense(512, activation='relu', input_dim=train_images_scaled.shape[1]),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compilarea modelului
mlp_model_images.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Antrenarea modelului
history_images = mlp_model_images.fit(
    train_images_scaled, train_labels,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    verbose=2
)

# Evaluarea modelului
loss_images, accuracy_images = mlp_model_images.evaluate(test_images_scaled, test_labels)
print(f"Test Loss (Images): {loss_images}, Test Accuracy (Images): {accuracy_images}")

# Afișarea rapoartelor de performanță
predictions_images = mlp_model_images.predict(test_images_scaled)
y_pred_images = predictions_images.argmax(axis=1)
print("Classification Report (Images):\n", classification_report(test_labels, y_pred_images))

# Matricea de confuzie
cm_images = confusion_matrix(test_labels, y_pred_images)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_images, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for MLP on Images')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fashion_mnist_images/confusion_matrix_images_mlp.png')

# Graficul pentru cerința 3.2
plt.figure(figsize=(10, 6))

# Curbele de eroare pentru imagini liniare
plt.plot(history_images.history['loss'], label='Train Loss (Images)', linestyle='-', marker='o')
plt.plot(history_images.history['val_loss'], label='Validation Loss (Images)', linestyle='-', marker='x')

# Curbele de acuratețe pentru imagini liniare
plt.plot(history_images.history['accuracy'], label='Train Accuracy (Images)', linestyle='-', marker='s')
plt.plot(history_images.history['val_accuracy'], label='Validation Accuracy (Images)', linestyle='-', marker='d')

plt.title('Loss and Accuracy Curves for MLP on Images')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('fashion_mnist_images/curves_images.png')
plt.show()


#================= CERINȚA 3.3 ==================
#Arhitectura CNN fără augmentări
cnn_model_no_aug = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

# Compilarea modelului
cnn_model_no_aug.compile(
    optimizer=RMSprop(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Antrenarea modelului fără augmentări
history_no_aug = cnn_model_no_aug.fit(
    train_images_cnn, train_labels,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    verbose=2
)

# Evaluarea modelului fără augmentări
loss_no_aug, accuracy_no_aug = cnn_model_no_aug.evaluate(test_images_cnn, test_labels)
print(f"Test Loss (No Augmentations): {loss_no_aug}, Test Accuracy (No Augmentations): {accuracy_no_aug}")

# ================== RAPORT FINAL ==================
# Afișarea rapoartelor de performanță
predictions_no_aug = cnn_model_no_aug.predict(test_images_cnn)
y_pred_no_aug = predictions_no_aug.argmax(axis=1)
print("Classification Report (CNN No Aug):\n", classification_report(test_labels, y_pred_no_aug))

# Matricea de confuzie
cm_no_aug = confusion_matrix(test_labels, y_pred_no_aug)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_no_aug, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for CNN (No Aug)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fashion_mnist_images/confusion_matrix_cnn_no_aug.png')


# Arhitectura CNN cu augmentări
data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.2,
    rescale=1.0/255.0
)
data_gen.fit(train_images_cnn)

cnn_model_with_aug = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

# Compilarea modelului
cnn_model_with_aug.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Antrenarea modelului cu augmentări
history_with_aug = cnn_model_with_aug.fit(
    data_gen.flow(train_images_cnn, train_labels, batch_size=128),
    validation_data=(test_images_cnn, test_labels),
    epochs=50,
    verbose=2
)

# Evaluarea modelului cu augmentări
loss_with_aug, accuracy_with_aug = cnn_model_with_aug.evaluate(test_images_cnn, test_labels)
print(f"Test Loss (With Augmentations): {loss_with_aug}, Test Accuracy (With Augmentations): {accuracy_with_aug}")

# Afișarea rapoartelor de performanță
predictions_with_aug = cnn_model_with_aug.predict(test_images_cnn)
y_pred_with_aug = predictions_with_aug.argmax(axis=1)
print("Classification Report (CNN With Aug):\n", classification_report(test_labels, y_pred_with_aug))

# Matricea de confuzie
cm_with_aug = confusion_matrix(test_labels, y_pred_with_aug)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_with_aug, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for CNN (With Aug)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fashion_mnist_images/confusion_matrix_cnn_with_aug.png')


# ================== GRAFICE ====================
# Grafic pentru loss
plt.figure(figsize=(12, 8))

# Curbele de eroare pentru modelul fără augmentări
plt.plot(history_no_aug.history['loss'], label='Train Loss (No Aug)', linestyle='--', marker='o')
plt.plot(history_no_aug.history['val_loss'], label='Validation Loss (No Aug)', linestyle='--', marker='x')

# Curbele de acuratețe pentru modelul fără augmentări
plt.plot(history_no_aug.history['accuracy'], label='Train Accuracy (No Aug)', linestyle='-', marker='o')
plt.plot(history_no_aug.history['val_accuracy'], label='Validation Accuracy (No Aug)', linestyle='-', marker='x')

plt.title('Loss Curves for CNN Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('fashion_mnist_images/curves_cnn_no.png')

# Grafic pentru acuratețe
plt.figure(figsize=(12, 8))



# Curbele de acuratețe pentru modelul cu augmentări
plt.plot(history_with_aug.history['accuracy'], label='Train Accuracy (With Aug)', linestyle='--', marker='s')
plt.plot(history_with_aug.history['val_accuracy'], label='Validation Accuracy (With Aug)', linestyle='--', marker='d')

# Curbele de eroare pentru modelul cu augmentări
plt.plot(history_with_aug.history['loss'], label='Train Loss (With Aug)', linestyle='-', marker='s')
plt.plot(history_with_aug.history['val_loss'], label='Validation Loss (With Aug)', linestyle='-', marker='d')

plt.title('Accuracy Curves for CNN Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('fashion_mnist_images/curves_cnn_aug.png')

# ================= CERINȚA 3.4 ==================
# Pregătirea datelor pentru ResNet-50
train_images_resnet = tf.image.resize(train_images_cnn, (32, 32))
train_images_resnet = tf.image.grayscale_to_rgb(train_images_resnet)
test_images_resnet = tf.image.resize(test_images_cnn, (32, 32))
test_images_resnet = tf.image.grayscale_to_rgb(test_images_resnet)

# Încărcarea ResNet-50 pre-antrenat
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Congelarea straturilor pre-antrenate
base_model.trainable = False

# Adăugarea straturilor personalizate
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

# Crearea modelului complet
resnet_model = Model(inputs=base_model.input, outputs=outputs)

# Compilarea modelului
resnet_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Antrenarea straturilor finale
history_resnet = resnet_model.fit(
    train_images_resnet, train_labels,
    validation_data=(test_images_resnet, test_labels),
    epochs=10,
    batch_size=64,
    verbose=2
)

# Deblocarea straturilor pentru fine-tuning
base_model.trainable = True

# Recompilarea modelului
resnet_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning
history_finetune = resnet_model.fit(
    train_images_resnet, train_labels,
    validation_data=(test_images_resnet, test_labels),
    epochs=10,
    batch_size=64,
    verbose=2
)

# Evaluarea modelului
loss_resnet, accuracy_resnet = resnet_model.evaluate(test_images_resnet, test_labels)
print(f"Test Loss (ResNet-50 Fine-tuning): {loss_resnet}, Test Accuracy (ResNet-50 Fine-tuning): {accuracy_resnet}")

# ================== GRAFICE ====================
# Grafic pentru loss
plt.figure(figsize=(12, 8))
plt.plot(history_resnet.history['loss'] + history_finetune.history['loss'], label='Train Loss', linestyle='--', marker='o')
plt.plot(history_resnet.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss', linestyle='-', marker='x')
plt.title('Loss Curves for Fine-tuned ResNet-50')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('fashion_mnist_images/resnet_loss.png')

# Grafic pentru acuratețe
plt.figure(figsize=(12, 8))
plt.plot(history_resnet.history['accuracy'] + history_finetune.history['accuracy'], label='Train Accuracy', linestyle='--', marker='s')
plt.plot(history_resnet.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='d')
plt.title('Accuracy Curves for Fine-tuned ResNet-50')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('fashion_mnist_images/resnet_accuracy.png')
plt.show()

plt.figure(figsize=(12, 8))

# Axe pentru Loss
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(history_resnet.history['loss'] + history_finetune.history['loss'], label='Train Loss', linestyle='--', marker='o', color='tab:red')
ax1.plot(history_resnet.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss', linestyle='-', marker='x', color='tab:orange')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')
ax1.grid(True)

# Axe pentru Accuracy (pe aceleași axe X, dar diferite Y)
ax2 = ax1.twinx()  # Creează al doilea ax împărțind același ax X
ax2.set_ylabel('Accuracy', color='tab:blue')
ax2.plot(history_resnet.history['accuracy'] + history_finetune.history['accuracy'], label='Train Accuracy', linestyle='--', marker='s', color='tab:blue')
ax2.plot(history_resnet.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='d', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

# Titlu și salvare
plt.title('Loss and Accuracy Curves for Fine-tuned ResNet-50')
plt.savefig('fashion_mnist_images/resnet_combined_loss_accuracy.png')
plt.show()

# ================ RAPORT FINAL =======================

# Afișarea rapoartelor de performanță
predictions_resnet = resnet_model.predict(test_images_resnet)
y_pred_resnet = predictions_resnet.argmax(axis=1)
print("Classification Report (ResNet-50):\n", classification_report(test_labels, y_pred_resnet))

# Matricea de confuzie
cm_resnet = confusion_matrix(test_labels, y_pred_resnet)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for ResNet-50')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fashion_mnist_images/confusion_matrix_resnet.png')

# Salvarea modelului
resnet_model.save('fashion_mnist_images/resnet_model.h5')
print("Modelul ResNet-50 a fost salvat cu succes!")



import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from collections import defaultdict
from skimage import io
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(subdir, file)
                label = os.path.basename(subdir)
                images.append(path)  # Asigurăm că adăugăm calea către fișier, nu fișierul încărcat
                labels.append(label)
    return images, labels


# Încărcarea imaginilor și etichetelor
training_images, train_labels = load_images_from_folder('fruits-360/Training')
print (f"Number of training images: {len(training_images)}")
print (f"Number of training labels: {len(train_labels)}")
test_images, test_labels = load_images_from_folder('fruits-360/Test')
print (f"Number of test images: {len(test_images)}")
print (f"Number of test labels: {len(test_labels)}")

# alegere 40 de clase random

random_classes = random.sample(sorted(list(set(train_labels))), 40)  # Convertește în listă și sortează
print("Randomly selected classes:\n", random_classes)

# Filtrarea imaginilor pentru cele 40 de clase selectate
selected_training_images = [img for img, lbl in zip(training_images, train_labels) if lbl in random_classes]
selected_training_labels = [lbl for lbl in train_labels if lbl in random_classes]
selected_test_images = [img for img, lbl in zip(test_images, test_labels) if lbl in random_classes]
selected_test_labels = [lbl for lbl in test_labels if lbl in random_classes]

print(f"Number of training images for selected classes: {len(selected_training_images)}")
print(f"Number of test images for selected classes: {len(selected_test_images)}")


# Preprocesare: Redimensionare și Normalizare
def preprocess_images(image_paths, target_size=(28, 28)):
    processed_images = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img_resized = img.resize(target_size)  # Redimensionare
        img_array = np.array(img_resized).astype('float32') / 255.0  # Normalizare
        processed_images.append(img_array.flatten())  # Aplatizare
    return np.array(processed_images)

# Preprocesare seturi selectate
X_train = preprocess_images(selected_training_images)
X_test = preprocess_images(selected_test_images)
y_train = np.array(selected_training_labels)
y_test = np.array(selected_test_labels)

print("Dimensiuni X_train:", X_train.shape)
print("Dimensiuni X_test:", X_test.shape)

#vizualizare rezultat pca
def plot_reconstructed_pca(X_train, X_train_pca, pca, idx=0):
    # Imaginea originală color
    img_color = X_train[idx].reshape(28, 28, 3)
    img_color = np.clip(img_color, 0, 1)  # Normalizare pentru afișare

    # Reconstruire imagine din PCA
    img_pca_reconstructed = pca.inverse_transform(X_train_pca[idx])  # Reconstruim din PCA
    img_pca_reconstructed = img_pca_reconstructed.reshape(28, 28, 3)
    img_pca_reconstructed = np.clip(img_pca_reconstructed, 0, 1)  # Normalizare pentru afișare

    plt.figure(figsize=(10, 5))
    # Imagine originală color
    plt.subplot(121)
    plt.imshow(img_color)
    plt.title('Original Image (Color)')
    plt.axis('off')

    # Imagine reconstruită din PCA
    plt.subplot(122)
    plt.imshow(img_pca_reconstructed)
    plt.title('Reconstructed Image (PCA)')
    plt.axis('off')
    plt.show()




# Extracția caracteristicilor HOG
def extract_hog_features_full(images, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9):
    hog_features = []
    for img in images:
        img_reshaped = img.reshape(28, 28, 3)  # Reconstrucție imagine color 3D
        img_grayscale = rgb2gray(img_reshaped)  # Conversie la grayscale (combină informația din R, G, B)
        features, _ = hog(img_grayscale, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, orientations=orientations, visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

# Aplicare HOG pe seturile de date
X_train_hog = extract_hog_features_full(X_train)
X_test_hog = extract_hog_features_full(X_test)

print("Dimensiuni X_train_hog:", X_train_hog.shape)
print("Dimensiuni X_test_hog:", X_test_hog.shape)


# Aplicare PCA
pca = PCA(n_components=50)  # Reducere la 50 de componente principale
X_train_pca = pca.fit_transform(X_train_hog)
X_test_pca = pca.transform(X_test_hog)

print("Dimensiuni X_train_pca:", X_train_pca.shape)
print("Dimensiuni X_test_pca:", X_test_pca.shape)


# afisare imagine hog
def plot_hog_image(X_train, X_train_hog, idx=0):
    img_color = X_train[idx].reshape(28, 28, 3)  # Imaginea originală color
    img_gray = rgb2gray(img_color)  # Conversie la grayscale pentru HOG
    _, hog_image = hog(img_gray, visualize=True)  # Obține imaginea HOG vizualizabilă

    plt.figure(figsize=(10, 5))
    # Imagine originală color
    plt.subplot(121)
    plt.imshow(img_color)
    plt.title('Original Image (Color)')
    plt.axis('off')

    # Imagine HOG
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Image (Visualized)')
    plt.axis('off')
    plt.show()


# Apelarea funcției pentru un exemplu din setul de antrenament
plot_hog_image(X_train, X_train_hog, idx=0)

# afisare iamgine pca
def plot_reconstructed_pca(X_train, X_train_pca, pca, idx=0):
    # Imaginea originală color
    img_color = X_train[idx].reshape(28, 28, 3)

    # Reconstruire imagine din PCA
    img_pca_reconstructed = pca.inverse_transform(X_train_pca[idx])  # Reconstruim din PCA
    img_pca_reconstructed = img_pca_reconstructed.reshape(36, 36)  # Dimensiunea originală color

    plt.figure(figsize=(10, 5))
    # Imagine originală color
    plt.subplot(121)
    plt.imshow(img_color)
    plt.title('Original Image (Color)')
    plt.axis('off')

    # Imagine reconstruită din PCA
    plt.subplot(122)
    plt.imshow(img_pca_reconstructed)
    plt.title('Reconstructed Image (PCA)')
    plt.axis('off')

    plt.show()

# Apelarea funcției pentru un exemplu din setul de antrenament
plot_reconstructed_pca(X_train, X_train_pca, pca, idx=0)

# reprezentare 2D a datelor din cele 40 de clase selectate

# Crearea unui DataFrame pentru stocarea caracteristicilor și a etichetelor
train_df = pd.DataFrame(X_train_pca)
train_df['label'] = y_train
test_df = pd.DataFrame(X_test_pca)
test_df['label'] = y_test

# Aplicarea PCA pentru reprezentarea 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_df.drop('label', axis=1))

# Vizualizarea PCA
plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=0,
    y=1,
    hue='label',  # Adaugă acest parametru pentru a colora punctele în funcție de etichete
    data=train_df,
    palette=sns.color_palette("hsv", n_colors=len(train_df['label'].unique()))  # Generază culori pentru fiecare clasă
)
plt.title('PCA plot of 40 random classes')
plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')  # Plasarea legendei în afara graficului
plt.show()
plt.savefig('pca_plot_fruits.png')

# distributia celor 40 de clase selectate
# Numărarea frecvenței fiecărei clase în setul de antrenament
class_distribution = defaultdict(int)

for label in y_train: # Iterăm prin fiecare etichetă
    class_distribution[label] += 1  # Incrementăm numărul de apariții al etichetei

# Crearea unui DataFrame pentru stocarea frecvenței claselor
class_distribution_df = pd.DataFrame(list(class_distribution.items()), columns=['Class', 'Frequency'])

# Vizualizarea distribuției claselor
plt.figure(figsize=(10, 5))
sns.barplot(x='Class', y='Frequency', data=class_distribution_df)
plt.title('Class Distribution of 40 Random Classes')
plt.xticks(rotation=45)
plt.show()
plt.savefig('class_distribution_fruits.png')

# Gradul de varianță cumulativă explicată de numărul ales de componente principale

pca_full = PCA()
pca_full.fit(X_train_hog)

plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of PCA')
plt.show()
plt.savefig('cumulative_variance_fruits.png')


print("Variance explained: ", np.sum(pca_full.explained_variance_ratio_[:50]))

# Impartirea datelor

X_train, X_val, y_train, y_val = train_test_split(train_df.drop('label', axis=1), train_df['label'])

# Standardizare

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Aplicarea selecției de caracteristici

selector = SelectPercentile(percentile=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)

# Compararea datelor inițiale cu cele standardizate și cu eliminarea varianței

print(f"Dimensiunea datelor inițiale: {X_train.shape}")
print(f"Dimensiunea datelor după standardizare: {X_train_scaled.shape}")
print(f"Dimensiunea datelor după standardizare și eliminarea varianței: {X_train_selected.shape}")

# impartirea datelor pentru algoritmii de clasificare

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)

# clasificare logistic regression

test_fold = np.concatenate([np.full(X_train.shape[0], -1), np.zeros(X_val.shape[0])])

# crearea obiectului PredefinedSplit

ps = PredefinedSplit(test_fold)

lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter  = 5000))
param_grid = {
    'logisticregression__C': np.logspace(-4, 4, 20),
    'logisticregression__multi_class': ['ovr', 'multinomial']
}

# Grid Search cu PredefinedSplit
grid_search = GridSearchCV(lr_pipeline, param_grid, cv=ps, verbose=2, n_jobs=-1)
grid_search.fit(np.vstack((X_val, X_train)), np.concatenate((y_val, y_train)))

# Afișarea rezultatelor
print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
print("Classification Report:\n", classification_report(y_val, grid_search.predict(X_val)))
print("Accuracy Score:", accuracy_score(y_val,grid_search.predict(X_val)))
print("Confusion Matrix:\n", confusion_matrix(y_val, grid_search.predict(X_val)))

# Afișarea matricei de confuzie
cm = confusion_matrix(y_val, grid_search.predict(X_val))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Matrice de Confuzie pentru Logistic Regression')
plt.xlabel('Predicții')
plt.ylabel('Valori Reale')
plt.savefig('matrice_confuzie_logistic_fruits.png')

svm_pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))

# Definirea unui grid mai restrâns de parametri
param_grid_svm = {
    'svc__C': np.logspace(-2, 2, 5),  # Valori mai restrânse
    'svc__kernel': ['linear', 'rbf']  # Focalizare pe două tipuri principale de kernel
}

# Configurarea GridSearchCV
grid_search_svm = GridSearchCV(svm_pipeline, param_grid_svm, cv=3, verbose=2, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)  # Folosirea unui subset reprezentativ dacă este necesar

# Afișarea rezultatelor
print("Best parameters for SVM:", grid_search_svm.best_params_)
print("Best score for SVM:", grid_search_svm.best_score_)
print("Classification Report:\n", classification_report(y_val, grid_search_svm.predict(X_val)))
print("Accuracy Score:", accuracy_score(y_val, grid_search_svm.predict(X_val)))

# Afișarea matricei de confuzie
cm = confusion_matrix(y_val, grid_search_svm.predict(X_val))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Matrice de Confuzie pentru SVM')
plt.xlabel('Predicții')
plt.ylabel('Valori Reale')
plt.savefig('confusion_matrix_svm_fruits.png')

# antrenarea unui clasificator Random Forest

X_combined = np.vstack((X_train, X_val))
y_combined = np.concatenate((y_train, y_val))

# Prepararea vectorului pentru PredefinedSplit
# Alocăm -1 pentru indicii din setul de validare și 0 pentru cei din setul de antrenament
test_fold = np.concatenate([
    -np.ones(X_val.shape[0], dtype=int),  # Setul de validare
    np.zeros(X_train.shape[0], dtype=int)  # Setul de antrenament
])

# Crearea obiectului PredefinedSplit
model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100],  # Mai puțini arbori
    'max_depth': [None, 10, 20],  # Limitarea adâncimii
    'min_samples_split': [4, 10],  # Creșterea numărului minim de eșantioane necesare pentru a împărți un nod
    'min_samples_leaf': [2, 6]  # Creșterea numărului minim de eșantioane necesare la o frunză
}

# Grid Search pentru a găsi cei mai buni hiperparametri
grid_search = GridSearchCV(model, param_grid_rf, cv=3, verbose=2, n_jobs=-1)  # Reducere CV și paralelizare completă
grid_search.fit(X_train, y_train)

# Evaluarea modelului pe setul de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
print("Best parameters:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Accuracy Score:", accuracy_score(y_val, y_pred))

# Afișarea matricei de confuzie
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_rf_fruits.png')



# antrenarea unui clasificator Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],  # Reducerea numărului de arbori
    'learning_rate': [0.01, 0.1],  # Ajustarea ratei de învățare
    'max_depth': [3, 5],  # Limitarea adâncimii arborilor
    'min_samples_split': [4, 10],  # Creșterea numărului minim de eșantioane necesare pentru a împărți un nod
    'min_samples_leaf': [2, 6]  # Creșterea numărului minim de eșantioane necesare la o frunză
}

# Grid Search pentru a găsi cei mai buni hiperparametri
grid_search = GridSearchCV(gb, param_grid, cv=3, verbose=2, n_jobs=-1)  # Reducere CV și paralelizare completă
grid_search.fit(X_train, y_train)

# Evaluarea modelului pe setul de validare
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
print("Best parameters for Gradient Boosting:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Accuracy Score:", accuracy_score(y_val, y_pred))

# Afișarea matricei de confuzie
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix for Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_gb_fruits.png')














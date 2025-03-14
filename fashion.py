
from sklearn.decomposition import PCA
from skimage.feature import hog
# from skimage.feature import hog
# from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import gzip
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import sys
sys.path.append('fashion-mnist/utils')  # Ajustează această cale la structura ta de directoare
from mnist_reader import load_mnist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.ndimage import zoom
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit


data_directory = '/fashion-mnist/data/fashion'

# Încărcarea datelor de antrenament
train_images, train_labels = load_mnist('fashion-mnist/data/fashion', kind='train')



# Incarcarea datelor de testare
test_images, test_labels = load_mnist('fashion-mnist/data/fashion', kind='t10k')



print("Dimensiuni imagini de antrenament:", train_images.shape)
print("Dimensiuni etichete de antrenament:", train_labels.shape)


# Preprocesare imagini
train_images = train_images.reshape((train_images.shape[0], -1))  # Transformă în vectori liniari
train_images = train_images.astype('float32') / 255  # Normalizare

# Aplicarea PCA pentru a reduce dimensionalitatea imaginilor
pca = PCA(n_components=50)  # Reduce la 50 de componente principale
X_train_pca = pca.fit_transform(train_images)
X_test_pca = pca.transform(test_images)


# Reconstrucția imaginilor din componentele PCA
train_images_reconstructed = pca.inverse_transform(X_train_pca).reshape(-1, 28, 28)
test_images_reconstructed = pca.inverse_transform(X_test_pca).reshape(-1, 28, 28)

# Extragerea caracteristicilor HOG
def extract_hog_image(image, pixels_per_cell=(4, 4), cells_per_block=(1, 1), orientations=8):
    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True)
    return hog_image

# Selectează o imagine de exemplu
index = 0
original_image = train_images[index].reshape(28, 28)
reconstructed_image = train_images_reconstructed[index]
hog_image = extract_hog_image(original_image)

# Crearea subploturilor
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image, cmap='gray', interpolation='bilinear')
axes[0].set_title('Imagine Originală')
axes[0].axis('off')

axes[1].imshow(reconstructed_image, cmap='gray', interpolation='bilinear')
axes[1].set_title('Imagine Reconstruită după PCA')
axes[1].axis('off')

axes[2].imshow(hog_image, cmap='gray')
axes[2].set_title('Vizualizare HOG')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('hog.png')

# Numărarea frecvenței fiecărei clase în setul de antrenament
plt.figure(figsize=(10, 5))
sns.countplot(x=train_labels)
plt.title('Distribuția Claselor în Setul de Antrenament')
plt.xlabel('Clase')
plt.ylabel('Număr de Imagini')
plt.savefig('distributie_clase_antrenament.png')

# Repetă pentru setul de testare dacă este necesar
plt.figure(figsize=(10, 5))
sns.countplot(x=test_labels)
plt.title('Distribuția Claselor în Setul de Testare')
plt.xlabel('Clase')
plt.ylabel('Număr de Imagini')
plt.savefig('distributie_clase_test.png')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=train_labels, palette='viridis', legend='full')
plt.title('Reprezentarea 2D a datelor Fashion-MNIST după PCA')
plt.xlabel('Componenta Principală 1')
plt.ylabel('Componenta Principală 2')
plt.legend(title='Clase', labels=np.unique(train_labels))
plt.savefig('reprezentare_2d_pca.png')

# afisare cateva imagini rezultrate in urma PCA
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Imaginea originală
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(train_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imaginea reconstruită
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(train_images_reconstructed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('reprezentare_pca.png')

# Calculul varianței explicative cumulativă
pca_full = PCA(n_components=50)
pca_full.fit(train_images)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("Varianța explicativă cumulativă:\n", cumulative_variance)
# Plot pentru varianța explicativă cumulativă
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Numărul de componente')
plt.ylabel('Varianța explicativă cumulativă')
plt.title('Varianța explicativă cumulativă folosind PCA')
plt.grid(True)
plt.savefig('varianta_explicativa_cumulativa.png')

# standardizarea datelor
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

# afisare dimensiune date standardizate
print("Dimensiuni imagini de antrenament standardizate:", train_images_scaled.shape)
print("Dimensiuni imagini de testare standardizate:", test_images_scaled.shape)

# calculul mediei și deviației standard
mean_train = np.mean(train_images_scaled)
std_train = np.std(train_images_scaled)
print("Media datelor de antrenament:", mean_train)
print("Deviația standard a datelor de antrenament:", std_train)

# calculul mediei și deviației standard
mean_test = np.mean(test_images_scaled)
std_test = np.std(test_images_scaled)
print("Media datelor de testare:", mean_test)
print("Deviația standard a datelor de testare:", std_test)

# afisare date standardizate
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Imaginea originală
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(train_images_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imaginea standardizată
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(test_images_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('standardizare_date.png')

selector_percentile = SelectPercentile(f_classif, percentile=10)
X_train_percentile = selector_percentile.fit_transform(train_images_scaled, train_labels)
X_test_percentile = selector_percentile.transform(test_images_scaled)

# afisare dimensiune date dupa selectia percentila
print("Dimensiuni imagini de antrenament după selecția percentilă:", X_train_percentile.shape)
print("Dimensiuni imagini de testare după selecția percentilă:", X_test_percentile.shape)



print("Numărul de atribute după selecția percentilă:", X_train_percentile.shape[1])

X_train, X_val, y_train, y_val = train_test_split(X_train_percentile, train_labels, test_size=0.2, random_state=42)

# impartirea datelor in set de antrenament si set de validare
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train_percentile, train_labels, test_size=0.2, random_state=42)

# Prepararea vectorului pentru PredefinedSplit
# Alocăm -1 pentru indicii din setul de validare și 0 pentru cei din setul de antrenament
test_fold = np.concatenate([
    -np.ones(X_val_sub.shape[0], dtype=int),  # Setul de validare
    np.zeros(X_train_sub.shape[0], dtype=int)  # Setul de antrenament
])

# Crearea obiectului PredefinedSplit
ps = PredefinedSplit(test_fold)

# Configurarea pipeline-ului și a parametrilor pentru GridSearch
lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
param_grid = {
    'logisticregression__C': np.logspace(-4, 4, 20),
    'logisticregression__multi_class': ['ovr', 'multinomial']
}

# Grid Search cu PredefinedSplit
grid_search = GridSearchCV(lr_pipeline, param_grid, cv=ps, verbose=2, n_jobs=-1)
grid_search.fit(np.vstack((X_val_sub, X_train_sub)), np.concatenate((y_val_sub, y_train_sub)))

# Afișarea rezultatelor
print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
print("Classification Report:\n", classification_report(y_val_sub, grid_search.predict(X_val_sub)))
print("Accuracy Score:", accuracy_score(y_val_sub, grid_search.predict(X_val_sub)))
print("Confusion Matrix:\n", confusion_matrix(y_val_sub, grid_search.predict(X_val_sub)))

# Afișarea matricei de confuzie
cm = confusion_matrix(y_val, grid_search.predict(X_val))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Matrice de Confuzie pentru Logistic Regression')
plt.xlabel('Predicții')
plt.ylabel('Valori Reale')
plt.savefig('matrice_confuzie_logistic.png')

# antrenarea unui clasificator SVM

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
cm = confusion_matrix(y_val_sub, grid_search_svm.predict(X_val_sub))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Matrice de Confuzie pentru SVM')
plt.xlabel('Predicții')
plt.ylabel('Valori Reale')
plt.savefig('confusion_matrix_svm.png')

# antrenarea unui clasificator Random Forest

X_combined = np.vstack((X_train_sub, X_val_sub))
y_combined = np.concatenate((y_train_sub, y_val_sub))

# Prepararea vectorului pentru PredefinedSplit
# Alocăm -1 pentru indicii din setul de validare și 0 pentru cei din setul de antrenament
test_fold = np.concatenate([
    -np.ones(X_val_sub.shape[0], dtype=int),  # Setul de validare
    np.zeros(X_train_sub.shape[0], dtype=int)  # Setul de antrenament
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
plt.savefig('confusion_matrix_rf.png')



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
plt.savefig('confusion_matrix_gb.png')

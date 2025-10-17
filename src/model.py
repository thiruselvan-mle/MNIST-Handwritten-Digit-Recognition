import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings 
warnings.filterwarnings('ignore')
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def split_x_y(df_train, df_test):
    X_train = df_train.drop('label', axis=1).values
    Y_train = df_train['label'].values
    X_test = df_test.drop('label', axis=1).values
    Y_test = df_test['label'].values
    return X_train, Y_train, X_test, Y_test

def normalize_data(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test

def reshape_for_cnn(X_train, X_test):
    X_train_cnn = X_train.reshape(-1,28,28,1)
    X_test_cnn = X_test.reshape(-1,28,28,1)
    return X_train_cnn, X_test_cnn

def encode_data(Y_train, Y_test):
    Y_train_cnn = to_categorical(Y_train,10)
    Y_test_cnn = to_categorical(Y_test,10)
    return Y_train_cnn, Y_test_cnn

def save_clean_data(save_dir, data_dict):
    os.makedirs(save_dir, exist_ok=True)

    for name, array in data_dict.items():
        path = os.path.join(save_dir, f"{name}.npy")
        np.save(path, array)
        print(f"✅ Saved {name}.npy  →  shape: {array.shape}")

    print(f"\nAll processed datasets saved in: {save_dir}")


def load_numpy_data(path, name=None):
    data = np.load(path)
    if name:
        print(f"{name} shape:", data.shape)
    return data

def train_val_split(X_train, Y_train, X_train_cnn, Y_train_cnn):
    X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(X_train_cnn, Y_train_cnn, test_size=0.2, random_state=42)
    return X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp, X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn

def mlp_train(X_train_mlp, y_train_mlp, X_val_mlp, y_val_mlp):
    mlp_model = Sequential([
        Dense(256, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mlp_history = mlp_model.fit(X_train_mlp, y_train_mlp, validation_data=(X_val_mlp, y_val_mlp), epochs=10, batch_size=128, verbose=1)
    return mlp_model, mlp_history

def cnn_train(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn):
    cnn_model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_history = cnn_model.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=10, batch_size=128, verbose=1)
    return cnn_model, cnn_history 

def compare_model_accu(mlp_history, cnn_history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(mlp_history.history['accuracy'], label='MLP Train')
    plt.plot(mlp_history.history['val_accuracy'], label='MLP Val')
    plt.title("MLP Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(cnn_history.history['accuracy'], label='CNN Train')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Val')
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def save_model(model, path, filename):
    os.makedirs(filename, exist_ok=True)

    model.save(os.path.join(path, filename))
    print("Model Successfully Saved")

def load_models(path,filename):
    model = load_model(os.path.join(path, filename))
    print("Model Loaded successfully")
    return model 

def evaluate_mlp(X_test, Y_test, mlp_model):
    mlp_test_proba = mlp_model.predict(X_test)
    mlp_test_pred = np.argmax(mlp_test_proba, axis=1)
    mlp_test_acc = accuracy_score(Y_test, mlp_test_pred)
    print(f"\nMLP Test Accuracy: {mlp_test_acc*100:.2f}")
    return mlp_test_pred, mlp_test_acc

def plot_mlp_cm(Y_test, mlp_test_pred):
    cm_mlp = confusion_matrix(Y_test, mlp_test_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - MLP Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def mlp_classification(Y_test, mlp_test_pred):
    print(classification_report(Y_test, mlp_test_pred))

def evaluate_cnn(X_test_cnn, Y_test_cnn, cnn_model):
    cnn_test_pred_prob = cnn_model.predict(X_test_cnn)
    cnn_test_pred = np.argmax(cnn_test_pred_prob, axis=1)
    y_true_cnn = np.argmax(Y_test_cnn, axis=1)
    cnn_test_acc = accuracy_score(y_true_cnn, cnn_test_pred)
    print(f"\nCNN Test Accuracy: {cnn_test_acc*100:.2f}")
    return y_true_cnn,  cnn_test_pred

def plot_cnn_cm(y_true_cnn, cnn_test_pred):
    cm_cnn = confusion_matrix(y_true_cnn, cnn_test_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix - CNN Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def cnn_classification(y_true_cnn, cnn_test_pred):
    print(classification_report(y_true_cnn, cnn_test_pred))

def plot_cnn_misclassified(cnn_test_pred, y_true_cnn, X_test_cnn):
    misclassified_idx = np.where(cnn_test_pred != y_true_cnn)[0]
    plt.figure(figsize=(12,6))
    for i, idx in enumerate(random.sample(list(misclassified_idx), 10)):
        plt.subplot(2,5,i+1)
        plt.imshow(X_test_cnn[idx].reshape(28,28), cmap='gray')
        plt.title(f"True:{y_true_cnn[idx]} Pred:{cnn_test_pred[idx]}")
        plt.axis("off")
    plt.suptitle("Misclassified Digits - CNN Test Set", fontsize=15)
    plt.show()
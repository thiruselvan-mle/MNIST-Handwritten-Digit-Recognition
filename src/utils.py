import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def distribution_plot(Y_train):
    plt.figure(figsize=(8,5))
    sns.countplot(x=Y_train, color='green')
    plt.title("Distribution of Digits (0–9)")
    plt.xlabel("Digit Label")
    plt.ylabel("Count")
    plt.show()

def sample_img(X_train, Y_train):
    fig, axes = plt.subplots(2, 5, figsize=(12,5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i].reshape(28,28), cmap="gray")
        ax.set_title(f"Label: {Y_train[i]}")
        ax.axis("off")
    plt.suptitle("Sample Digit Images", fontsize=16)
    plt.show()

def plot_intensity(X_train):
    plt.figure(figsize=(8,5))
    plt.hist(X_train.ravel(), bins=30, color="blue", alpha=0.7)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value (0–1 after normalization)")
    plt.ylabel("Frequency")
    plt.show()


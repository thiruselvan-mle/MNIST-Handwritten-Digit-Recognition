# 🧠 MNIST Handwritten Digit Recognition
A Deep Learning project built using CNN & Streamlit
This project predicts handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
Users can draw digits directly on a canvas or upload images to see instant predictions through a Streamlit web app.

---

<img src=app/digit.png width=100% height=600>

## 📘 Project Overview
Handwritten digit recognition is one of the classic problems in computer vision and deep learning.
The goal is to correctly identify digits (0–9) from images of handwritten numbers.

This project:

 - Uses the MNIST dataset (28×28 grayscale images)

 - Trains models using Deep Learning (CNN) and optionally Machine Learning (MLP)

 - Deploys an interactive web app using Streamlit
---

## 📂 Project Structure
```bash
MNIST-Handwritten-Digit-Recognition/
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned/normalized data 
│
├── models/
│   ├── cnn_model.keras     # Trained CNN model
│   └── mlp_model.keras    # Trained MLP model 
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-data-cleaning.ipynb
│   ├── 03-eda.ipynb
│   ├── 04-model-training.ipynb
│   └── 05-model-evaluation.ipynb
│
├── src/
│   └── (Python files for data processing & training)
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation
```

## 🧩 Dataset Information

Source: Kaggle MNIST Dataset ( https://www.kaggle.com/datasets/oddrationale/mnist-in-csv )

Description:

 - Each image is 28×28 pixels

 - Grayscale intensity (0–255)

 - label column represents the digit (0–9)

Example:
```bash
label	1x1	 1x2	...	28x28
5	     0	  0	    ...	  0
0	     0	 255	...	  0
```

## ⚙️ Installation & Setup

1️⃣ Clone this Repository
```bash 
 git clone https://github.com/thiruselvan-mle/MNIST-Handwritten-Digit-Recognition.git
 cd MNIST-Handwritten-Digit-Recognition
```

2️⃣ Create Virtual Environment (optional but recommended)
```bash
 python -m venv venv
 # On Windows
 venv\Scripts\activate    
 # On Mac/Linux    
 source venv/bin/activate     
```

3️⃣ Install Dependencies
```bash
 pip install -r requirements.txt

 Or manually:

 pip install streamlit tensorflow joblib opencv-python pillow streamlit-drawable-canvas numpy
```
## 🚀 Running the Application
```bash
  streamlit run app.py

  Then open the local URL (e.g., http://localhost:8501) in your browser.
```

## 🖼️ App Features
### 🖊️ Draw Digit

Use your mouse or touchscreen to draw digits (0–9).

Click “Predict” to see model output.

### 📤 Upload Digit Image

 - Upload any 28×28 grayscale image or larger.
 -  
 - App preprocesses and predicts automatically.

## 🧠 Model Information
```bash
Model	|     Type	     |      File	     |  Accuracy
CNN	    | Deep Learning	 |   cnn_model.keras |   ~99%
MLP	    |Machine Learning|	mlp_model.keras  | 	 ~97%

The CNN model is the default for deployment.
```

## 📊 Example Prediction Output
🧠 Predicted Digit: 7

<img src=app/demo.png width=100% height=600>


## 📈 Notebooks Summary
```bash
### Notebook	     |          Description
01-data-exploration	 | Loaded and explored the dataset structure
02-data-cleaning	 | Handled missing data and normalization
03-eda	Visualized   | digit patterns using Seaborn & Matplotlib
04-model-training	 | Trained CNN & MLP models
05-model-evaluation	 | Compared model performance and metrics
```

## 🧾 Requirements

 - Python 3.9+

 - TensorFlow

 - Streamlit

 - OpenCV

 - NumPy

 - Pillow

 - Joblib

 - streamlit-drawable-canvas


## 💡 Future Improvements

 - Add more complex CNN architectures (LeNet, ResNet)

 - Deploy on cloud (Streamlit Cloud / Hugging Face)

 - Add data augmentation

 - Create REST API endpoint for model predictions

## 👨‍💻 Author

 Thiruselvan MUthuraman

  - 🔗 MNIST Digit Recognition Project
  - 💬 “Turning data into intelligent applications!”
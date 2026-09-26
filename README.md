````markdown
# AI Deepfake Image Detection Using Explainable AI

## 📌 Project Overview

This project presents an **AI-based Deepfake Image Detection system using Explainable Artificial Intelligence (XAI)**.

The system uses a deep learning **XceptionNet CNN model** to classify facial images as either **Real** or **Fake**. In addition to prediction, the system provides visual explanations using **Grad-CAM, LIME, and SHAP** to help understand which regions of an image influenced the model's decision.

The trained model is developed in **Google Colab** and the final application is implemented using **Streamlit**. **GitHub** is used for source-code and project version management.

---

## 🎯 Objectives

- Detect whether an uploaded facial image is Real or Fake.
- Use XceptionNet for deepfake image classification.
- Provide prediction confidence.
- Explain model predictions using Explainable AI techniques.
- Visualize important image regions using Grad-CAM.
- Generate local explanations using LIME.
- Analyze feature contributions using SHAP.
- Develop an easy-to-use Streamlit web application.

---

## 🧠 Proposed System

The proposed system follows this workflow:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Resize to 299 × 299
     ↓
XceptionNet
     ↓
Real / Fake Classification
     ↓
Confidence Score
     ↓
Explainable AI
 ┌─────────┬─────────┬─────────┐
 ↓         ↓         ↓
Grad-CAM  LIME      SHAP
 └─────────┴─────────┴─────────┘
     ↓
Visual Explanation
     ↓
Streamlit Interface
````

---

## 📊 Dataset

The project uses a dataset containing real and fake facial images.

### Dataset Classes

| Class | Description                               |
| ----- | ----------------------------------------- |
| Fake  | AI-generated or manipulated facial images |
| Real  | Authentic facial images                   |

The original dataset contains:

* **1,500 Fake images**
* **1,501 Real images**
* **3,001 images in total**

For experiments with a smaller dataset, a balanced subset of **1,000 images** can also be used:

* 500 Fake
* 500 Real

The dataset is divided into:

```text
70% Training
15% Validation
15% Testing
```

> The exact dataset size and split used for the final reported experiment should be stated in the project results.

---

## 🏗️ Model Architecture

The project uses **XceptionNet**, a convolutional neural network architecture that is particularly useful for image classification.

The model consists of:

```text
Input Image
    ↓
XceptionNet
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Layer
    ↓
Dropout
    ↓
Sigmoid Output
    ↓
Real / Fake
```

### Input Size

```text
299 × 299 × 3
```

### Output

```text
0 → Fake
1 → Real
```

The final classification threshold is approximately:

```text
0.5
```

---

## 🔍 Explainable AI

A major feature of this project is the use of Explainable AI.

### 1. Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** creates a heatmap showing the image regions that contributed strongly to the model's prediction.

Example:

```text
Original Image
      ↓
XceptionNet
      ↓
Grad-CAM
      ↓
Important Image Regions
```

Grad-CAM helps users visually understand where the model focused.

---

### 2. LIME

**Local Interpretable Model-Agnostic Explanations (LIME)** explains an individual prediction by modifying parts of an image and observing how the prediction changes.

LIME highlights image regions that have a strong influence on a particular prediction.

---

### 3. SHAP

**SHapley Additive exPlanations (SHAP)** provides feature-attribution information to explain how different parts of an input contribute to the model output.

SHAP is used to provide an additional explanation of the model's prediction.

---

## 💻 Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming                    |
| Google Colab | Model development and training |
| TensorFlow   | Deep learning                  |
| Keras        | Neural network implementation  |
| XceptionNet  | Image classification           |
| OpenCV       | Image processing               |
| NumPy        | Numerical computation          |
| Scikit-learn | Model evaluation               |
| Grad-CAM     | Visual explanation             |
| LIME         | Local explanation              |
| SHAP         | Feature attribution            |
| Streamlit    | Web application                |
| GitHub       | Version control                |

---

## 📁 Project Structure

```text
deepfake-xai-detection/
│
├── app.py
├── deepfake_xai_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

If the model is managed using Git LFS:

```text
deepfake-xai-detection/
│
├── app.py
├── deepfake_xai_model.keras
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-xai-detection.git
```

Move into the project directory:

```bash
cd deepfake-xai-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The main Python libraries used are:

```text
tensorflow
streamlit
numpy
pillow
opencv-python-headless
lime
shap
scikit-image
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser.

Usually:

```text
http://localhost:8501
```

---

## 🖥️ Application Workflow

### Step 1

Upload a facial image.

### Step 2

The image is resized to:

```text
299 × 299
```

### Step 3

The image is passed to the trained XceptionNet model.

### Step 4

The system predicts:

```text
REAL
```

or

```text
FAKE
```

### Step 5

The confidence score is displayed.

### Step 6

Explainable AI methods provide visual explanations:

```text
Grad-CAM
LIME
SHAP
```

---

## 📈 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example evaluation:

```text
                    Predicted
                 Fake      Real

Actual Fake        TP        FN
Actual Real        FP        TN
```

The final accuracy should be reported based on the performance on the **held-out test set**.

---

## 🔬 Why Explainable AI?

Traditional deepfake detection systems often provide only:

```text
Prediction: Fake
```

This project attempts to provide additional information:

```text
Prediction: Fake
Confidence: XX%

Explanation:
The highlighted regions are the areas that
contributed most strongly to the model's prediction.
```

This makes the system easier to inspect and understand.

Importantly, an XAI heatmap indicates **model attribution**, not proof that a specific highlighted region contains a particular manipulation artifact.

---

## 🌐 Deployment

The application can be deployed using Streamlit-compatible hosting.

Deployment workflow:

```text
Google Colab
     ↓
Train XceptionNet
     ↓
Save Model
     ↓
GitHub
     ↓
Streamlit
     ↓
Web Application
```

---

## 🚀 Future Enhancements

Future versions of the project can include:

* Video deepfake detection.
* Real-time webcam analysis.
* Larger and more diverse datasets.
* Face detection and face cropping before classification.
* Temporal deepfake detection for videos.
* Improved XAI visualizations.
* Ensemble deep learning models.
* Real-time content verification.
* Integration with digital forensics tools.

---

## 🎓 Applications

Potential applications include:

* Social media content verification
* Digital forensics
* Cybersecurity
* Journalism
* Online media verification
* Research and education
* Fake-content detection

---

## ⚠️ Limitations

* Performance depends on the quality and diversity of the training dataset.
* A model trained on a limited dataset may not generalize to every type of deepfake.
* XAI explanations describe model behavior and should not be treated as definitive forensic proof.
* Detection confidence does not guarantee that an image is genuinely real or fake.

---

## 👩‍💻 Project Type

**Academic / Student AI-ML Project**

### Domain

```text
Artificial Intelligence
Machine Learning
Deep Learning
Computer Vision
Explainable AI
```

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👤 Author

**Suriya S**

AI / ML Student Project

---

## ⭐ Acknowledgement

This project was developed using open-source machine learning and explainability technologies including TensorFlow, Keras, XceptionNet, Grad-CAM, LIME, SHAP, and Streamlit.

````

### Your GitHub folder should now look like this

```text
deepfake-xai-detection
│
├── app.py
├── deepfake_xai_model.keras
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
````


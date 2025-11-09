# 💧 Leak Detection in Urban Water Distribution Network

## 🧠 Project Overview
**Leak Detection in Urban Water Distribution Network** is a Machine Learning-based web system designed to detect and predict potential leaks within city water supply pipelines. The project integrates **data-driven anomaly detection**, **statistical feature analysis**, and **interactive visualizations** to enhance the efficiency of water management systems.

This application allows users to:
- Upload water flow and pressure datasets.
- Train ML models for leak detection.
- View detailed analytics, feature importance, and performance metrics.
- Make real-time predictions on new data samples.

---

## 🌐 Tech Stack

| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Backend (API)** | FastAPI (Python) |
| **Machine Learning** | Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn |
| **Visualization** | Dynamic charts (ROC, Confusion Matrix, Heatmap, etc.) |
| **Deployment** | Localhost / Cloud-based FastAPI server |

---

## 🚀 Key Features

### 1. 📁 Dataset Upload & Training
- Upload CSV datasets directly from the web interface.
- Automatically performs:
  - Data preprocessing and normalization.
  - Feature correlation and visualization.
  - Train-test split and model evaluation.

### 2. 📊 Interactive Results Dashboard
- Displays:
  - Accuracy, Precision, Recall, F1-score.
  - Confusion matrix.
  - ROC and Precision-Recall curves.
  - Feature importance plots and correlations.

### 3. 🔮 Leak Prediction Interface
- Real-time prediction by entering sensor readings (flow rate, pressure, etc.).
- Outputs leak probability with confidence values.

### 4. 🧩 Model Insights
- Visual insights via dynamically generated plots:
  - Histograms
  - Heatmaps
  - Feature importance
  - ROC and PR Curves

---

## ⚙️ Installation & Setup

### 🔸 Prerequisites
Ensure you have the following installed:
- Python 3.9+
- pip
- FastAPI
- Node.js (optional for UI serving)

### 🔸 Clone the Repository
```bash
git clone https://github.com/your-username/Leak-Detection-in-Urban-Water-Distribution-Network.git
cd Leak-Detection-in-Urban-Water-Distribution-Network

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import io
import base64
from typing import Dict, Union, List
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_names = []
training_results = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global model, feature_names, training_results
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        possible_targets = [c for c in df.columns if "leak" in c.lower() or "status" in c.lower()]
        if possible_targets:
            target = possible_targets[0]
        else:
            binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
            if not binary_cols:
                raise HTTPException(status_code=400, detail="No leak/status column found")
            target = binary_cols[0]
        
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = le.fit_transform(df[col].astype(str))
        
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        params = {
            'n_estimators':[100,200],
            'max_depth':[5,10,20,None],
            'min_samples_split':[2,5],
            'min_samples_leaf':[1,2]
        }
        
        grid = GridSearchCV(RandomForestClassifier(), params, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        model = grid.best_estimator_
        feature_names = X.columns.tolist()
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for idx, col in enumerate(numeric_cols[:4]):
            axes[idx//2, idx%2].hist(df[col], bins=20)
            axes[idx//2, idx%2].set_title(col)
        plt.tight_layout()
        hist_img = fig_to_base64(fig1)
        
        fig2 = plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        corr_img = fig_to_base64(fig2)
        
        fig3 = plt.figure(figsize=(8, 5))
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        feat_img = fig_to_base64(fig3)
        
        fig4 = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_img = fig_to_base64(fig4)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig5 = plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_img = fig_to_base64(fig5)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig6 = plt.figure(figsize=(6, 4))
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        pr_img = fig_to_base64(fig6)
        
        training_results = {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "best_params": grid.best_params_,
            "feature_names": feature_names,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "images": {
                "histogram": hist_img,
                "correlation": corr_img,
                "feature_importance": feat_img,
                "confusion_matrix": cm_img,
                "roc_curve": roc_img,
                "precision_recall": pr_img
            }
        }
        
        return JSONResponse(content=training_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "API Running"}

@app.get("/features")
def get_features():
    if not feature_names:
        raise HTTPException(status_code=400, detail="No model trained yet")
    return {"features": feature_names}

@app.get("/results")
def get_results():
    if not training_results:
        raise HTTPException(status_code=400, detail="No training results available")
    return JSONResponse(content=training_results)

class PredictionRequest(BaseModel):
    features: Dict[str, Union[float, int, str]]

@app.post("/predict")
def predict(request: PredictionRequest):
    global model, feature_names
    
    if model is None:
        raise HTTPException(status_code=400, detail="No model trained yet")
    
    try:
        features_list = []
        for name in feature_names:
            if name not in request.features:
                raise HTTPException(status_code=400, detail=f"Missing feature: {name}")
            
            value = request.features[name]
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {name}")
            
            features_list.append(value)
        
        features_array = np.array(features_list).reshape(1, -1)
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)[0]
        
        return {
            "prediction": int(prediction[0]),
            "probability": {
                "no_leak": float(probability[0]),
                "leak": float(probability[1])
            },
            "status": "success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
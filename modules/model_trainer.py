import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 简单的模型训练器，负责数据切分和预处理
class ModelTrainer:
    def __init__(self):
        self.data_path = "data/raw/clinical_data_raw.csv"
        self.feature_path = "data/meta/selected_features.json"
        self.output_path = "data/processed/train_test_data.pkl"
        self.modeling_data_path = "data/processed/modeling_data.csv"
        os.makedirs("data/processed", exist_ok=True)

    def run(self):
        print("\n>> [Model Trainer] Starting training pipeline...")
        
        # 1. Load Data
        df = pd.read_csv(self.data_path)
        if 'ID' in df.columns: df = df.drop(columns=['ID'])
        df = df.fillna(df.mean()) # 简单填补
        
        # 2. Load Features
        with open(self.feature_path, 'r') as f:
            features = json.load(f)['final_features']
        
        print(f"   Using {len(features)} features: {features}")
        
        # 3. Prepare X, y
        target = 'Outcome'
        X = df[features]
        y = df[target]
        
        # 保存一份用于建模的干净数据
        df_modeling = df[features + [target]]
        df_modeling.to_csv(self.modeling_data_path, index=False)
        
        # 4. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 5. Preprocessing (Standardization)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
        
        # 6. SMOTE
        print("   Applying SMOTE to training set...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        
        # 7. Save
        data_dict = {
            'X_train_smote': X_train_smote,
            'y_train_smote': y_train_smote,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'scaler': scaler
        }
        joblib.dump(data_dict, self.output_path)
        print(f"✅ Data processed and saved to {self.output_path}")

if __name__ == "__main__":
    ModelTrainer().run()

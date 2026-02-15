import json
import pandas as pd
import os

# R 输出的特征列表 (从你的 log 中复制)
r_features = ['Age', 'GCS', 'Hunthess', 'Modified.fisher', 'Wfns', 'PNI', 'PAR', 'NLPR', 'SIRI', 'SII', 'AISI']

print(">> Bridging R features to Python dataset...")
raw_path = "data/raw/clinical_data_raw.csv"
if not os.path.exists(raw_path):
    print(f"❌ Error: {raw_path} not found.")
    exit(1)

df = pd.read_csv(raw_path)
py_cols = df.columns.tolist()

# 映射逻辑: R的 "Modified.fisher" -> Python的 "Modified fisher"
final_features = []
for rf in r_features:
    if rf in py_cols:
        final_features.append(rf)
    else:
        # 尝试把点换成空格
        rf_space = rf.replace('.', ' ')
        if rf_space in py_cols:
            final_features.append(rf_space)
        else:
            print(f"⚠️ Warning: R feature '{rf}' could not be mapped to Python columns!")

print(f"✅ Successfully mapped {len(final_features)} features:")
print(final_features)

# 保存到 meta
os.makedirs("data/meta", exist_ok=True)
with open("data/meta/selected_features.json", "w") as f:
    json.dump({"source": "R_Kaggle_Reproduction", "final_features": final_features}, f, indent=4)
print(">> data/meta/selected_features.json updated.")

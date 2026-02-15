
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

st.set_page_config(layout="wide", page_title="aSAH 风险预测")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

st.markdown("""<style>
.result-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 20px; }
.metric-value { font-size: 32px; font-weight: bold; color: #333; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model_resources():
    try:
        data_dict = joblib.load("data/processed/train_test_data.pkl")
        with open("data/meta/selected_features.json", "r") as f:
            feats = json.load(f)["final_features"]
        X = data_dict["X_train_smote"][feats]
        y = data_dict["y_train_smote"]
        model = XGBClassifier(eval_metric="logloss", random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        return model, feats, explainer
    except: return None, None, None

model, features, explainer = load_model_resources()

if model is None:
    st.error("🚨 模型加载失败，请检查数据文件。")
    st.stop()

with st.sidebar:
    st.header("📝 临床数据录入")
    opt_gcs = {1: "13-15分 (轻度)", 2: "8-12分 (中度)", 3: "3-7分 (重度)"}
    opt_wfns = {1: "I-III 级", 2: "IV-V 级"}
    opt_fisher = {1: "I-II 级", 2: "III-IV 级"}
    opt_hunt = {1: "I-II 级", 2: "III-V 级"}
    opt_pni = {1: "< 40.7 (极差)", 2: "40.7-45.0", 3: "45.0-48.8", 4: "> 48.8 (良好)"}
    
    with st.expander("1. 核心评分", expanded=True):
        gcs = st.selectbox("GCS", [1,2,3], format_func=lambda x: opt_gcs[x])
        wfns = st.selectbox("WFNS", [1,2], format_func=lambda x: opt_wfns[x])
        fisher = st.selectbox("Fisher", [1,2], format_func=lambda x: opt_fisher[x])
        hunt = st.selectbox("Hunt-Hess", [1,2], format_func=lambda x: opt_hunt[x])

    with st.expander("2. 实验室指标", expanded=True):
        pni = st.selectbox("PNI", [1,2,3,4], format_func=lambda x: opt_pni[x])
        alb = st.selectbox("Albumin", [1,2,3,4])
        siri = st.selectbox("SIRI", [1,2,3,4])
        sii = st.selectbox("SII", [1,2,3,4])
        nar = st.selectbox("NAR", [1,2,3,4])
        plr = st.selectbox("PLR", [1,2,3,4])
    
    with st.expander("3. 基础信息", expanded=False):
        age = st.selectbox("Age", [1,2,3,4])
        aneurysm = st.radio("位置", [1, 2], format_func=lambda x: "前循环" if x==1 else "后循环")
        surgery = st.radio("手术", [1, 2], format_func=lambda x: "介入" if x==1 else "夹闭")
        htn = st.checkbox("高血压")

    input_dict = {feat: 2 for feat in features}
    user_inputs = {"GCS": gcs, "Wfns": wfns, "Hunthess": hunt, "Modified fisher": fisher, "PNI": pni, "Albumin": alb, "SIRI": siri, "SII": sii, "NAR": nar, "PLR": plr, "Age": age, "Aneurysm location": aneurysm, "Surgical method": surgery, "Hypertension": 1 if htn else 0}
    for k,v in user_inputs.items(): 
        if k in features: input_dict[k] = v
    input_df = pd.DataFrame([input_dict])
    predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

st.title("🧠 临床决策支持系统 (CDSS)")

if predict_btn:
    prob = float(model.predict_proba(input_df)[0][1])
    if prob < 0.3: risk, color = "低风险", "green"
    elif prob < 0.7: risk, color = "中风险", "orange"
    else: risk, color = "高风险", "red"

    st.markdown(f"""<div class='result-card' style='border-left-color: {color};'>
    <div><span class='metric-value' style='color:{color}'>{prob*100:.1f}%</span> <span style='font-size:20px'>({risk})</span></div>
    </div>""", unsafe_allow_html=True)

    st.subheader("🔍 关键因素")
    c1, c2 = st.columns([3, 2])
    shap_vals = explainer.shap_values(input_df)[0]
    
    with c2:
        fi = pd.DataFrame({"F": input_df.columns, "S": shap_vals, "V": input_df.iloc[0]}).sort_values(by="S", key=abs, ascending=False).head(5)
        for _, r in fi.iterrows():
            icon = "🔺" if r["S"] > 0 else "🔽"
            st.write(f"{icon} **{r['F']}** (值:{int(r['V'])})")
            
    with c1:
        try:
            fig = plt.figure(figsize=(10, 3))
            shap.force_plot(explainer.expected_value, shap_vals, input_df.iloc[0], matplotlib=True, show=False)
            st.pyplot(fig, bbox_inches="tight")
        except: st.write("绘图加载中...")
else:
    st.info("👈 请在左侧点击预测")

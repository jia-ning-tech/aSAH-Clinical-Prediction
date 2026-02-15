import streamlit as st
import base64
import os

st.set_page_config(
    page_title="aSAH 预后预测复现",
    page_icon="🧠",
    layout="wide"
)

# --- 核心优化：图片转 Base64 嵌入 (解决隧道加载慢的问题) ---
def img_to_html(img_path, width="100%"):
    if not os.path.exists(img_path):
        return f"<p>图片文件未找到: {img_path}</p>"
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width}; border-radius: 5px;">'

st.title("🧠 动脉瘤性蛛网膜下腔出血 (aSAH) 短期预后预测")
st.markdown("#### 基于机器学习的文献复现项目 (BMC Medicine 2026)")

st.info("💡 提示：所有图片已内嵌优化，即使在慢速网络下也能立即显示。")

# 图片展示区
st.subheader("📊 复现结果展示")

tab1, tab2, tab3, tab4 = st.tabs(["ROC 曲线 (性能)", "校准曲线 (准确度)", "DCA (临床获益)", "SHAP (可解释性)"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(img_to_html("app/assets/roc_curves.png"), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        ### 分析解读
        - **RF (随机森林)** 在训练集 CV 中表现最佳 (AUC ~0.91)。
        - **LR (逻辑回归)** 在独立测试集中表现最稳健 (AUC ~0.79)。
        - **结论**：复现结果验证了 GCS、WFNS 等指标的强预测价值。
        """)

with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(img_to_html("app/assets/calibration_curve.png"), unsafe_allow_html=True)
    with col2:
        st.info("校准曲线展示了模型预测概率与真实发生率的一致性。大部分模型在中风险区间表现良好。")

with tab3:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(img_to_html("app/assets/dca_curve.png"), unsafe_allow_html=True)
    with col2:
        st.info("DCA 曲线显示，在阈值 0.1-0.5 之间，使用本模型指导临床决策能带来正向的净获益。")

with tab4:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(img_to_html("app/assets/shap_summary.png"), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        ### 关键因子 (Top Features)
        1. **GCS & WFNS**: 评分越高(病情越重)，风险越高。
        2. **PNI**: 营养指数越高，风险越低 (保护因素)。
        3. **炎症指标**: SIRI/SII 升高与不良预后相关。
        """)

st.divider()
st.caption("项目路径: /workspace/jn-神经外科001 | 复现者: User & Gemini Agent")

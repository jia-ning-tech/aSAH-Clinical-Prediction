import subprocess
import os
import sys
import time

# ================= CONFIGURATION =================
# 你想用哪种方式进行特征筛选？
# "PYTHON" = 使用 feature_selector.py (V6.0 纯 Python 复现 Boruta/LASSO)
# "R_BRIDGE" = 使用 bridge_r_features.py (读取你在 Kaggle R 语言跑出的结果)
FEATURE_MODE = "R_BRIDGE"  # <--- 修改这里来切换模式！建议您现在设为 R_BRIDGE
# =================================================

def print_header(step_name):
    print("\n" + "="*60)
    print(f"🚀 [Auto-Pilot] {step_name}")
    print("="*60)

def run_command(command, description):
    print(f"\n>> Status: Running {description}...")
    start_time = time.time()
    
    try:
        # 运行命令并在出错时抛出异常
        result = subprocess.run(command, shell=True, check=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ Success! (Time: {elapsed:.2f}s)")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error detected in step: {description}")
        print(f"   Command failed: {command}")
        print("   Pipeline stopped.")
        sys.exit(1)

def main():
    print_header("Initializing Project Pipeline")
    print(f"   Mode: {FEATURE_MODE}")
    print(f"   Root: {os.getcwd()}")
    
    # --- Step 1: 特征筛选 ---
    print_header("Step 1: Feature Selection")
    if FEATURE_MODE == "PYTHON":
        run_command("python modules/feature_selector.py", "Python Boruta + LASSO Selection")
    elif FEATURE_MODE == "R_BRIDGE":
        # 确保桥接脚本存在
        if not os.path.exists("modules/bridge_r_features.py"):
            print("❌ Missing modules/bridge_r_features.py!")
            print("   Please run the previous step to generate the bridge script.")
            sys.exit(1)
        run_command("python modules/bridge_r_features.py", "Bridging R Features to Python")
    else:
        print(f"❌ Unknown mode: {FEATURE_MODE}")
        sys.exit(1)

    # --- Step 2: 模型训练 ---
    print_header("Step 2: Model Training")
    run_command("python modules/model_trainer.py", "Retraining 6 ML Models")

    # --- Step 3: 报表生成 ---
    print_header("Step 3: Generating Publication Reports")
    run_command("python modules/publication_reporter.py", "Generating SCI Tables & Figures")

    # --- Summary ---
    print_header("Pipeline Completed Successfully")
    print("🎉 All tasks finished! You can find your results here:")
    print(f"   📂 Charts & Tables:  {os.path.join(os.getcwd(), 'results')}")
    print(f"   📂 App Assets:       {os.path.join(os.getcwd(), 'app/assets')}")
    print("\n   Next Step: Run 'python -m streamlit run app/main.py' to view the web app.")

if __name__ == "__main__":
    main()

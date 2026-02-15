import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# 绘图风格设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class FeatureSelector:
    def __init__(self, raw_data_path):
        self.output_dir = "results"
        self.meta_dir = "data/meta"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        
        print(">> Loading data...")
        df = pd.read_csv(raw_data_path)
        if 'ID' in df.columns: df = df.drop(columns=['ID'])
        df = df.fillna(df.mean())
        
        self.target_col = 'Outcome'
        self.X = df.drop(columns=[self.target_col])
        self.y = df[self.target_col]
        self.feature_names = np.array(self.X.columns)
        
        # 标准化 (LASSO 必须)
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
        self.X_np = self.X_scaled.values
        self.y_np = self.y.values

    # ==========================================================================
    # Boruta 核心逻辑
    # ==========================================================================
    def run_boruta(self):
        print("\n>> [Step 1] Running Boruta (Simulating Iterations)...")
        
        # 1. 获取最终分类结果
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=100)
        boruta.fit(self.X_np, self.y_np)
        
        # 分类特征
        feat_conf = self.feature_names[boruta.support_]
        feat_tent = self.feature_names[boruta.support_weak_]
        feat_rej = self.feature_names[~(boruta.support_ | boruta.support_weak_)]
        
        print(f"   Status: {len(feat_conf)} Confirmed, {len(feat_tent)} Tentative, {len(feat_rej)} Rejected")

        # 2. 模拟迭代以获取箱线图数据 (Shadow History)
        n_iter = 60
        history_real = {f: [] for f in self.feature_names}
        history_shadow = {'S_Min': [], 'S_Mean': [], 'S_Max': []}
        
        rf_sim = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
        
        print("   Generating history distribution...")
        for _ in range(n_iter):
            # 生成 Shadow
            X_shadow = np.apply_along_axis(np.random.permutation, 0, self.X_np)
            X_boruta = np.hstack([self.X_np, X_shadow])
            rf_sim.fit(X_boruta, self.y_np)
            
            imps = rf_sim.feature_importances_
            n_real = self.X_np.shape[1]
            real_imps = imps[:n_real]
            shadow_imps = imps[n_real:]
            
            # 记录 Real
            for i, name in enumerate(self.feature_names):
                history_real[name].append(real_imps[i])
            
            # 记录 Shadow Stats
            history_shadow['S_Min'].append(np.min(shadow_imps))
            history_shadow['S_Mean'].append(np.mean(shadow_imps))
            history_shadow['S_Max'].append(np.max(shadow_imps))

        # 3. 组装绘图数据 (顺序：Shadow -> Rejected -> Tentative -> Confirmed)
        plot_data = []
        plot_labels = []
        plot_colors = []
        
        # 3.1 Shadow (Blue)
        for k in ['S_Min', 'S_Mean', 'S_Max']:
            plot_data.append(history_shadow[k])
            plot_labels.append(k.replace('_', '-')) # S-Min
            plot_colors.append('#2b83ba') # Blue
            
        # 3.2 Rejected (Red) - 按中位数排序
        if len(feat_rej) > 0:
            medians = [np.median(history_real[f]) for f in feat_rej]
            sorted_idx = np.argsort(medians)
            for i in sorted_idx:
                f = feat_rej[i]
                plot_data.append(history_real[f])
                plot_labels.append(f)
                plot_colors.append('#d7191c') # Red

        # 3.3 Tentative (Yellow)
        if len(feat_tent) > 0:
            medians = [np.median(history_real[f]) for f in feat_tent]
            sorted_idx = np.argsort(medians)
            for i in sorted_idx:
                f = feat_tent[i]
                plot_data.append(history_real[f])
                plot_labels.append(f)
                plot_colors.append('#fdae61') # Yellow/Orange

        # 3.4 Confirmed (Green)
        if len(feat_conf) > 0:
            medians = [np.median(history_real[f]) for f in feat_conf]
            sorted_idx = np.argsort(medians)
            for i in sorted_idx:
                f = feat_conf[i]
                plot_data.append(history_real[f])
                plot_labels.append(f)
                plot_colors.append('#1a9641') # Green

        # 4. 绘图 Fig 3A
        plt.figure(figsize=(14, 8))
        bp = plt.boxplot(plot_data, patch_artist=True, labels=plot_labels, showfliers=False)
        
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_linewidth(1)
            
        plt.title("Fig 3A. Boruta Feature Importance", fontsize=16, fontweight='bold')
        plt.xticks(rotation=90, ha='center', fontsize=10)
        plt.ylabel("Importance (Z-Score)", fontsize=12)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2b83ba', label='Shadow'),
            Patch(facecolor='#d7191c', label='Rejected'),
            Patch(facecolor='#fdae61', label='Tentative'),
            Patch(facecolor='#1a9641', label='Confirmed')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig3A_Boruta.png")
        print(f"   Saved: {self.output_dir}/Fig3A_Boruta.png")
        
        self.confirmed_features = feat_conf.tolist()
        return self.confirmed_features

    # ==========================================================================
    # LASSO 核心逻辑 (glmnet style)
    # ==========================================================================
    def run_lasso(self, input_features):
        print(f"\n>> [Step 2] Running LASSO (glmnet-style) on {len(input_features)} features...")
        if len(input_features) == 0: return []
        
        X = self.X_scaled[input_features].values
        y = self.y_np
        n, p = X.shape
        
        # 1. 计算理论上的 Lambda_max (让所有系数归零的最小值)
        # 公式: ||X^T * (y - y_bar)||_inf / N
        # 对于 Logistic Regression，这个推导略复杂，但可以用线性近似起步
        # 这里的做法：计算梯度在 0 点的 L_inf 范数
        y_centered = y - y.mean()
        grad_0 = np.abs(np.dot(X.T, y_centered))
        lambda_max_theory = np.max(grad_0) / n
        
        # 稍微放大一点以确保万无一失 (glmnet 内部处理可能会 rescale)
        lambda_max = lambda_max_theory * 10.0
        # Lambda_min 通常设为 max 的 0.0001 倍
        lambda_min = lambda_max * 1e-4
        
        print(f"   Lambda Range: {lambda_max:.4f} -> {lambda_min:.4f}")
        
        # 生成 Log-spaced Lambdas (从大到小)
        n_alphas = 100
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_alphas)
        
        # 对应 sklearn C = 1 / lambda
        Cs = 1.0 / lambdas
        
        # 2. 计算路径 (Path)
        print("   Computing path coefficients...")
        coefs = []
        dfs = [] # Degrees of Freedom
        
        # 使用 liblinear (对小数据更稳，更容易画出平滑 path)
        # 虽然 saga 支持 elasticnet，但 liblinear L1 也是经典的 lasso
        # 我们手动 warm start
        clf = LogisticRegression(penalty='l1', solver='liblinear', 
                                 max_iter=10000, tol=1e-6, random_state=42)
        
        last_coef = None
        for c in Cs:
            clf.set_params(C=c)
            # 简单的 warm start 模拟 (liblinear 不支持显式 warm_start=True init，但收敛快)
            # 如果用 saga 可以 warm_start
            clf.fit(X, y)
            
            c_curr = clf.coef_[0].copy()
            coefs.append(c_curr)
            # 统计非零个数 (绝对值 > 1e-5)
            dfs.append(np.sum(np.abs(c_curr) > 1e-5))
            
        coefs = np.array(coefs) # shape (n_alphas, n_features)
        
        # 3. 交叉验证 (CV)
        print("   Running CV (metric: Deviance)...")
        # 使用 StratifiedKFold 手动做，确保 metrics 一致
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_deviances = np.zeros((5, n_alphas))
        
        for k, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            clf_cv = LogisticRegression(penalty='l1', solver='liblinear', 
                                        max_iter=10000, tol=1e-6, random_state=42)
            
            for i, c in enumerate(Cs):
                clf_cv.set_params(C=c)
                clf_cv.fit(X_tr, y_tr)
                # Predict proba
                prob = clf_cv.predict_proba(X_val)
                # Calc Deviance = 2 * log_loss * N
                ll = log_loss(y_val, prob, normalize=False) # Sum of log loss
                cv_deviances[k, i] = 2 * ll
                
        # 统计
        mean_dev = np.mean(cv_deviances, axis=0)
        std_dev = np.std(cv_deviances, axis=0)
        se_dev = std_dev / np.sqrt(5)
        
        # 4. 寻找 Min 和 1SE
        # 注意：lambdas 是从大到小 (Strong -> Weak)
        # 模型复杂度：简单(DF=0) -> 复杂(DF=p)
        # Deviance 通常是 U 型或下降型
        
        idx_min = np.argmin(mean_dev)
        min_dev = mean_dev[idx_min]
        target_se = min_dev + se_dev[idx_min]
        
        # 1SE Rule: 在 mean_dev <= target_se 的集合里，选最简单的模型
        # 最简单 = Lambda 最大 = index 最小
        candidates = np.where(mean_dev <= target_se)[0]
        idx_1se = np.min(candidates)
        
        val_lam_min = lambdas[idx_min]
        val_lam_1se = lambdas[idx_1se]
        
        # Log Lambda (自然对数)
        log_lambdas = np.log(lambdas)
        log_lam_min = np.log(val_lam_min)
        log_lam_1se = np.log(val_lam_1se)

        # ======================================================================
        # 绘图 Fig 3B (CV)
        # ======================================================================
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        ax.errorbar(log_lambdas, mean_dev, yerr=se_dev, 
                    fmt='o', color='#e41a1c', ecolor='#999999', 
                    elinewidth=1.0, capsize=2, markersize=3)
        
        ax.axvline(x=log_lam_min, linestyle='--', color='k', alpha=0.6, label='Min')
        ax.axvline(x=log_lam_1se, linestyle='--', color='b', alpha=0.6, label='1-SE')
        
        ax.set_xlabel('Log Lambda')
        ax.set_ylabel('Binomial Deviance')
        ax.set_title('Fig 3B. LASSO Cross-Validation', fontsize=14, fontweight='bold')
        ax.invert_xaxis() # glmnet 习惯：左边大Lambda(简单)，右边小Lambda(复杂)？
                          # 不，通常 x轴是 log(lambda)。
                          # 如果 x = log(lambda)，则 左边(负大)是小lambda，右边(正大)是大lambda
                          # 我们保持标准坐标轴：左小右大。
                          # 但 glmnet 的 df 轴习惯对齐 lambda
        
        # 添加顶部 DF 轴
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        # 选刻度
        ticks_pos = np.linspace(min(log_lambdas), max(log_lambdas), 8)
        ticks_labels = []
        for tp in ticks_pos:
            # 找最近的 index
            idx = np.abs(log_lambdas - tp).argmin()
            ticks_labels.append(str(dfs[idx]))
        ax_top.set_xticks(ticks_pos)
        ax_top.set_xticklabels(ticks_labels)
        ax_top.set_xlabel('Degrees of Freedom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig3B_LASSO_CV.png")
        print(f"   Saved: {self.output_dir}/Fig3B_LASSO_CV.png")

        # ======================================================================
        # 绘图 Fig 3C (Path)
        # ======================================================================
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        # 绘线
        # 只有最后系数非0的才标颜色，全0的画淡一点
        active_indices = np.where(np.abs(coefs[-1]) > 1e-4)[0]
        
        for i in range(p):
            if i in active_indices:
                ax.plot(log_lambdas, coefs[:, i], linewidth=1.5, alpha=0.9)
            else:
                # 那些一直被压缩的，画淡一点
                ax.plot(log_lambdas, coefs[:, i], color='grey', linewidth=0.5, alpha=0.3)
                
        ax.axvline(x=log_lam_min, linestyle='--', color='k', alpha=0.5)
        ax.axvline(x=log_lam_1se, linestyle='--', color='b', alpha=0.5)
        
        ax.set_xlabel('Log Lambda')
        ax.set_ylabel('Coefficients')
        ax.set_title('Fig 3C. LASSO Coefficient Profiles', fontsize=14, fontweight='bold')
        
        # 顶部 DF 轴
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(ticks_pos)
        ax_top.set_xticklabels(ticks_labels)
        ax_top.set_xlabel('Degrees of Freedom')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig3C_LASSO_Path.png")
        print(f"   Saved: {self.output_dir}/Fig3C_LASSO_Path.png")
        
        # 最终输出 (1se)
        final_feats = np.array(input_features)[np.abs(coefs[idx_1se]) > 1e-5].tolist()
        print(f"\n   ✅ Final Features (1se): {final_feats}")
        return final_feats

    def save_results(self, final_features):
        with open(f"{self.meta_dir}/selected_features.json", "w") as f:
            json.dump({"final_features": final_features}, f, indent=4)

if __name__ == "__main__":
    raw_data = "data/raw/clinical_data_raw.csv"
    if os.path.exists(raw_data):
        selector = FeatureSelector(raw_data)
        boruta_feats = selector.run_boruta()
        if len(boruta_feats) > 0:
            final_feats = selector.run_lasso(boruta_feats)
            selector.save_results(final_feats)

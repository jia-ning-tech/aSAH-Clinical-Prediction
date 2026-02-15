import numpy as np
import pandas as pd
from scipy import stats, optimize
from statsmodels.stats.outliers_influence import variance_inflation_factor

class QuantileFitter:
    @staticmethod
    def fit_lognorm(q1, q2, q3):
        def objective(params):
            s, scale = params
            if s <= 0 or scale <= 0: return 1e9
            loss = (stats.lognorm.ppf(0.25, s=s, scale=scale) - q1)**2 + \
                   (stats.lognorm.ppf(0.50, s=s, scale=scale) - q2)**2 + \
                   (stats.lognorm.ppf(0.75, s=s, scale=scale) - q3)**2
            return loss
        res = optimize.minimize(objective, [0.5, q2], method='Nelder-Mead', tol=1e-4)
        return res.x

    @staticmethod
    def fit_norm(q1, q2, q3):
        loc = q2
        scale = (q3 - q1) / 1.35
        return [loc, scale]

class DataSimulator:
    def __init__(self, configs):
        self.configs = configs
        self.variables = list(configs.keys())
        self.n_vars = len(self.variables)
        self.fitted_params = {}
        self._fit_all_distributions()

    def _fit_all_distributions(self):
        print(">> [Init] 正在拟合连续变量分布参数...")
        for name, cfg in self.configs.items():
            if cfg['type'] == 'continuous_bins':
                bps = cfg['breakpoints']
                q1, q2, q3 = bps[0], bps[1], bps[2]
                if cfg.get('default_dist') == 'lognorm':
                    params = QuantileFitter.fit_lognorm(q1, q2, q3)
                    self.fitted_params[name] = {'dist': 'lognorm', 'params': params}
                else:
                    params = QuantileFitter.fit_norm(q1, q2, q3)
                    self.fitted_params[name] = {'dist': 'norm', 'params': params}

    def _build_block_correlation_matrix(self):
        mat = np.eye(self.n_vars)
        vars_idx = {v: i for i, v in enumerate(self.variables)}
        
        def set_block_corr(var_list, corr_val):
            valid_vars = [v for v in var_list if v in vars_idx]
            for i in range(len(valid_vars)):
                for j in range(i+1, len(valid_vars)):
                    u, v = vars_idx[valid_vars[i]], vars_idx[valid_vars[j]]
                    mat[u, v] = corr_val
                    mat[v, u] = corr_val

        def set_pair_corr(v1, v2, val):
            if v1 in vars_idx and v2 in vars_idx:
                u, v = vars_idx[v1], vars_idx[v2]
                mat[u, v] = val
                mat[v, u] = val

        # === 1. 特征间相关性 (VIF控制) ===
        # 临床评分组
        set_pair_corr('GCS', 'Wfns', -0.6) 
        set_pair_corr('GCS', 'Hunthess', -0.55)
        set_pair_corr('Wfns', 'Hunthess', 0.65)
        set_pair_corr('Modified fisher', 'Wfns', 0.4)

        # 炎症组
        inflamm_group = ['SIRI', 'AISI', 'NLPR', 'SII', 'WBC', 'Neutrophil']
        set_block_corr(inflamm_group, 0.45) 
        
        # 营养组
        set_pair_corr('Albumin', 'PNI', 0.6)
        set_pair_corr('Albumin', 'NAR', -0.3)

        # === 2. 核心修正：特征与结局 (Outcome) 的相关性 ===
        # 注意：在 configs.py 中 Outcome=1 代表 Unfavorable (Bad)
        # GCS: 1=Good, 3=Bad -> 正相关 (越高越Bad)
        # WFNS: 1=Good, 2=Bad -> 正相关
        set_pair_corr('Outcome', 'GCS', 0.60)  # 强预测因子
        set_pair_corr('Outcome', 'Wfns', 0.55)
        set_pair_corr('Outcome', 'Hunthess', 0.50)
        set_pair_corr('Outcome', 'Modified fisher', 0.40)
        
        # 炎症指标: 通常越高越不好 -> 正相关
        set_pair_corr('Outcome', 'SIRI', 0.35)
        set_pair_corr('Outcome', 'AISI', 0.30)
        set_pair_corr('Outcome', 'SII', 0.30)
        set_pair_corr('Outcome', 'PNI', -0.30) # PNI越高营养越好 -> 负相关
        
        # 基础特征
        set_pair_corr('Outcome', 'Age', 0.25) # 年龄大风险高

        return mat

    def generate_data(self, n_samples=1120, random_state=42):
        np.random.seed(random_state)
        correlation_matrix = self._build_block_correlation_matrix()
        
        # 修复正定性
        min_eig = np.min(np.real(np.linalg.eigvals(correlation_matrix)))
        if min_eig < 0:
            correlation_matrix -= 1.1 * min_eig * np.eye(self.n_vars)

        mean = np.zeros(self.n_vars)
        Z = np.random.multivariate_normal(mean, correlation_matrix, n_samples)
        U = stats.norm.cdf(Z)
        
        df_raw = pd.DataFrame()
        df_binned = pd.DataFrame()

        for i, var_name in enumerate(self.variables):
            u_vec = U[:, i]
            cfg = self.configs[var_name]
            
            if cfg['type'] == 'continuous_bins':
                fit_info = self.fitted_params[var_name]
                params = fit_info['params']
                if fit_info['dist'] == 'lognorm':
                    raw_values = stats.lognorm.ppf(u_vec, s=params[0], scale=params[1])
                else:
                    raw_values = stats.norm.ppf(u_vec, loc=params[0], scale=params[1])
                
                raw_values = np.clip(raw_values, 
                                     stats.scoreatpercentile(raw_values, 0.1), 
                                     stats.scoreatpercentile(raw_values, 99.9))
                
                df_raw[var_name] = np.round(raw_values, 2)
                bps = [-np.inf] + cfg['breakpoints'] + [np.inf]
                df_binned[var_name] = pd.cut(raw_values, bins=bps, labels=[1, 2, 3, 4]).astype(int)
                
            elif cfg['type'] == 'ordinal' or cfg['type'] == 'categorical':
                probs = cfg['probs']
                cum_probs = np.cumsum(probs)
                mapping = cfg.get('mapping', {}) 
                # Ordinal 的 values 已经是 [1, 2] 这种
                vals = cfg['values'] if 'values' in cfg else range(len(probs))
                
                indices = np.searchsorted(cum_probs, u_vec)
                indices = np.clip(indices, 0, len(probs)-1)
                
                if 'values' in cfg:
                    final_val = [vals[x] for x in indices]
                    df_binned[var_name] = final_val
                    df_raw[var_name] = final_val
                else:
                    # Categorical: raw 存 0/1, binned 存 0/1 (或者按需调整)
                    # 这里 mapping {0:0, 1:1}
                    mapped_vals = [mapping.get(x, x) for x in indices]
                    df_binned[var_name] = mapped_vals
                    df_raw[var_name] = mapped_vals

        return df_raw, df_binned

    def calculate_vif(self, df):
        df_num = df.select_dtypes(include=[np.number])
        df_num = df_num.loc[:, df_num.std() > 0.01]
        vif = pd.DataFrame()
        vif["Variable"] = df_num.columns
        vif["VIF"] = [variance_inflation_factor(df_num.values, i) 
                      for i in range(df_num.shape[1])]
        return vif.sort_values(by="VIF", ascending=False)

if __name__ == "__main__":
    from configs import VARIABLE_CONFIGS
    sim = DataSimulator(VARIABLE_CONFIGS)
    print(">> [Re-Run] 生成具有强预测信号的数据...")
    df_raw, df_binned = sim.generate_data(n_samples=1120)
    
    df_raw.to_csv("data/raw/clinical_data_raw.csv", index=False)
    df_binned.to_csv("data/processed/modeling_data.csv", index=False)
    print("✅ 数据已更新 (注入了 Outcome 相关性)")

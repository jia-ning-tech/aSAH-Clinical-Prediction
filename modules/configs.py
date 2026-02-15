"""
配置文件：定义变量的统计分布特征
基于文献 Table 1 和 Table S1 数据
"""

# 核心配置：定义每个变量的类型和分布参数
# type='continuous_bins': 对应文献中的四分位数离散化 (Q1, Median, Q3)
# type='categorical': 对应文献中的 n(%)
VARIABLE_CONFIGS = {
    # --- 结局变量 ---
    'Outcome': {
        'type': 'categorical', 
        'probs': [0.6923, 0.3077],  # Favorable (0-2), Unfavorable (3-6)
        'mapping': {0: 0, 1: 1}     # 0=Good, 1=Poor
    },

    # --- 人口学特征 ---
    'Age': {
        'type': 'continuous_bins', 
        'breakpoints': [50, 58, 67], # Table 1: <50, 50-58, 58-67, >67
        'default_dist': 'norm'       # 年龄通常近似正态
    },
    'Sex': {
        'type': 'categorical', 
        'probs': [0.3748, 0.6252],   # Male 37.48%, Female 62.52%
        'mapping': {0: 1, 1: 2}      # 1=Male, 2=Female (假设编码)
    },
    'Smoke': {
        'type': 'categorical', 'probs': [0.7766, 0.2234], 'mapping': {0: 0, 1: 1} # No/Yes
    },
    'Drink': {
        'type': 'categorical', 'probs': [0.8167, 0.1833], 'mapping': {0: 0, 1: 1}
    },
    'Hypertension': {
        'type': 'categorical', 'probs': [0.4882, 0.5118], 'mapping': {0: 0, 1: 1}
    },
    'Diabets': {
        'type': 'categorical', 'probs': [0.9281, 0.0719], 'mapping': {0: 0, 1: 1}
    },

    # --- 临床评分 (Ordinal) ---
    # 注意：文献中 GCS 分为 3 组: 13-15(57.95%), 8-12(34.16%), 3-7(7.88%)
    # 我们将其编码为 1, 2, 3 (注意：GCS分数越高越好，但为了风险评分，通常反向或按严重度分组)
    # 这里按 Table 1 顺序：Group1(轻), Group2(中), Group3(重) -> 对应数值 1, 2, 3
    'GCS': {
        'type': 'ordinal', 
        'probs': [0.5795, 0.3416, 0.0788], 
        'values': [1, 2, 3] # 1=13-15分, 2=8-12分, 3=3-7分
    },
    'Hunthess': {
        'type': 'ordinal',
        'probs': [0.6189, 0.3811],
        'values': [1, 2] # 1=I-II级, 2=III-V级
    },
    'Modified fisher': {
        'type': 'ordinal',
        'probs': [0.6037, 0.3963],
        'values': [1, 2] # 1=I-II级, 2=III-IV级
    },
    'Wfns': {
        'type': 'ordinal',
        'probs': [0.6086, 0.3914],
        'values': [1, 2] # 1=I-III级, 2=IV-V级
    },
    'Aneurysm location': {
        'type': 'categorical',
        'probs': [0.8838, 0.1162], # Anterior, Posterior
        'mapping': {0: 1, 1: 2}
    },
    'Surgical method': {
        'type': 'categorical',
        'probs': [0.6466, 0.3534], # Interventional, Clipping
        'mapping': {0: 1, 1: 2}
    },

    # --- 实验室指标 (Continuous -> Quartiles) ---
    # 使用 Table 1 的分箱边界 (breakpoints)
    'Ddimer': {
        'type': 'continuous_bins', 'breakpoints': [0.5, 1.1, 2.6], 'default_dist': 'lognorm'
    },
    'Albumin': {
        'type': 'continuous_bins', 'breakpoints': [35.2, 39.5, 42.6], 'default_dist': 'norm'
    },
    'Fibrinogen': {
        'type': 'continuous_bins', 'breakpoints': [2.45, 2.95, 3.51], 'default_dist': 'norm'
    },
    'PNI': {
        'type': 'continuous_bins', 'breakpoints': [40.7, 45.0, 48.8], 'default_dist': 'norm'
    },
    'NAR': {
        'type': 'continuous_bins', 'breakpoints': [0.17, 0.24, 0.31], 'default_dist': 'lognorm'
    },
    'PAR': {
        'type': 'continuous_bins', 'breakpoints': [4.11, 5.11, 6.24], 'default_dist': 'lognorm'
    },
    'NLPR': {
        'type': 'continuous_bins', 'breakpoints': [0.03, 0.05, 0.08], 'default_dist': 'lognorm'
    },
    'PLR': {
        'type': 'continuous_bins', 'breakpoints': [133.7, 208, 295], 'default_dist': 'lognorm'
    },
    'MLR': {
        'type': 'continuous_bins', 'breakpoints': [0.33, 0.52, 0.88], 'default_dist': 'lognorm'
    },
    'SIRI': {
        'type': 'continuous_bins', 'breakpoints': [2.57, 5.08, 9.68], 'default_dist': 'lognorm'
    },
    'CLR': {
        'type': 'continuous_bins', 'breakpoints': [1.59, 4.81, 18.4], 'default_dist': 'lognorm'
    },
    'SII': {
        'type': 'continuous_bins', 'breakpoints': [1084.4, 1860.8, 3037.3], 'default_dist': 'lognorm'
    },
    'AISI': {
        'type': 'continuous_bins', 'breakpoints': [488.4, 979.0, 1863.1], 'default_dist': 'lognorm'
    },
    'Procalcitonin': {
        'type': 'continuous_bins', 'breakpoints': [0.04, 0.12, 0.43], 'default_dist': 'lognorm'
    },
    'Sugar': {
        'type': 'continuous_bins', 'breakpoints': [6.5, 7.4, 8.8], 'default_dist': 'norm'
    },
    'Creatinine': {
        'type': 'continuous_bins', 'breakpoints': [46, 55, 67], 'default_dist': 'norm'
    },
    'Hb': {
        'type': 'continuous_bins', 'breakpoints': [116.0, 128.0, 140.0], 'default_dist': 'norm'
    },
    'RBC': {
        'type': 'continuous_bins', 'breakpoints': [3.75, 4.15, 4.56], 'default_dist': 'norm'
    }
}

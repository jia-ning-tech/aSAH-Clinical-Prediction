| Dataset      | Model   |   AUC |   Accuracy |   Sensitivity |   Specificity |   Brier Score |
|:-------------|:--------|------:|-----------:|--------------:|--------------:|--------------:|
| Training Set | LR      | 0.893 |      0.792 |         0.806 |         0.778 |         0.133 |
| Training Set | SVM     | 0.95  |      0.889 |         0.925 |         0.853 |         0.086 |
| Training Set | GBM     | 0.992 |      0.949 |         0.962 |         0.936 |         0.05  |
| Training Set | NN      | 1     |      1     |         1     |         1     |         0     |
| Training Set | RF      | 1     |      1     |         1     |         1     |         0.014 |
| Training Set | XGBoost | 1     |      1     |         1     |         1     |         0.001 |
| Test Set     | LR      | 0.885 |      0.786 |         0.796 |         0.781 |         0.139 |
| Test Set     | SVM     | 0.872 |      0.774 |         0.713 |         0.803 |         0.143 |
| Test Set     | GBM     | 0.879 |      0.812 |         0.759 |         0.838 |         0.134 |
| Test Set     | NN      | 0.84  |      0.801 |         0.657 |         0.868 |         0.186 |
| Test Set     | RF      | 0.864 |      0.795 |         0.639 |         0.868 |         0.142 |
| Test Set     | XGBoost | 0.877 |      0.792 |         0.713 |         0.829 |         0.149 |
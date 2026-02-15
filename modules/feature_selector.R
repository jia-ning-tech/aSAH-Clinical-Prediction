# ==============================================================================
# R Script for BMC Medicine Reproduction (SCI Publication Quality)
# ==============================================================================

# 1. 安装与加载包
if(!require(Boruta)) install.packages("Boruta")
if(!require(glmnet)) install.packages("glmnet")
if(!require(caret)) install.packages("caret")

library(Boruta)
library(glmnet)
library(caret)

# 2. 读取数据 (确保路径正确)
df <- read.csv("/kaggle/input/asah-ml/clinical_data_raw.csv")

# 预处理
if("ID" %in% names(df)) df$ID <- NULL
# 简单填补 (中位数)
preProc <- preProcess(df, method = "medianImpute")
df <- predict(preProc, df)

# 定义 X 和 y
y <- df$Outcome
X <- df[, setdiff(names(df), "Outcome")]

# ==============================================================================
# Part A: Boruta 特征筛选 (SCI 风格优化)
# ==============================================================================
set.seed(42)
cat(">> Running Boruta...\n")
boruta_res <- Boruta(Outcome ~ ., data = df, doTrace = 2, maxRuns = 100)
final_boruta <- TentativeRoughFix(boruta_res)

# --- 绘图 Fig 3A: Boruta Importance ---
# 设置高分辨率输出 (300 DPI), 宽度 8英寸, 高度 6英寸
png("Fig3A_Boruta_SCI.png", width = 8, height = 6, units = "in", res = 300, family = "sans")

# 调整边距: 下边距留大一点给特征名 (bottom, left, top, right)
par(mar = c(8, 5, 2, 2)) 

# 自定义颜色: 绿色(Confirmed), 黄色(Tentative), 红色(Rejected), 蓝色(Shadow)
# 使用更柔和的学术配色
col_codes <- c("#2ca02c", "#ff7f0e", "#d62728", "#1f77b4")

# 绘图
# xaxt="n" 暂时不画x轴标签，后面手动加
# main="" 去掉图内标题 (论文规范)
plot(final_boruta, xlab = "", ylab = "Importance (Z-Score)", main = "", 
     las = 2, cex.axis = 0.8, col = col_codes, xaxt = "n")

# 手动添加 X 轴标签，防止重叠
lz <- lapply(1:ncol(final_boruta$ImpHistory), function(i)
  final_boruta$ImpHistory[is.finite(final_boruta$ImpHistory[,i]),i])
names(lz) <- colnames(final_boruta$ImpHistory)
Labels <- sort(sapply(lz, median))
axis(side = 1, las = 2, labels = names(Labels), 
     at = 1:ncol(final_boruta$ImpHistory), cex.axis = 0.9, font = 2) # font=2 加粗特征名

# 添加图例 (可选，如果正文有说明可不加，这里为了清晰加上)
legend("topleft", legend = c("Confirmed", "Tentative", "Rejected", "Shadow"), 
       fill = col_codes, bty = "n", cex = 0.8)

dev.off()
cat(">> Fig 3A Saved.\n")

# 提取特征
boruta_feats <- getSelectedAttributes(final_boruta, withTentative = FALSE)
cat("Boruta Selected:", boruta_feats, "\n")

# ==============================================================================
# Part B: LASSO 回归 (SCI 风格优化)
# ==============================================================================
# 准备矩阵
f <- as.formula(paste("Outcome ~", paste(boruta_feats, collapse = "+")))
x_lasso <- model.matrix(f, df)[,-1]
y_lasso <- as.factor(df$Outcome)

set.seed(42)
cv_fit <- cv.glmnet(x_lasso, y_lasso, family = "binomial", type.measure = "deviance", alpha = 1)

# --- 绘图 Fig 3B: LASSO CV (Deviance) ---
png("Fig3B_LASSO_CV_SCI.png", width = 6, height = 5, units = "in", res = 300, family = "sans")

# 调整边距
par(mar = c(5, 5, 4, 2))

# 绘图
# cex.lab: 坐标轴标题大小, cex.axis: 刻度文字大小
plot(cv_fit, xlab = "Log(Lambda)", ylab = "Binomial Deviance", 
     cex.lab = 1.2, cex.axis = 1.0, col = "#d62728", pch = 19) # 红点实心

# 强化辅助线 (Min 和 1se)
abline(v = log(cv_fit$lambda.min), lty = 2, lwd = 1.5, col = "black")
abline(v = log(cv_fit$lambda.1se), lty = 2, lwd = 1.5, col = "blue")

# 添加简明标注
text(log(cv_fit$lambda.min), max(cv_fit$cvm), "Min", pos = 3, cex = 0.8, col = "black")
text(log(cv_fit$lambda.1se), max(cv_fit$cvm), "1-SE", pos = 3, cex = 0.8, col = "blue")

dev.off()
cat(">> Fig 3B Saved.\n")

# --- 绘图 Fig 3C: LASSO Coefficient Profiles ---
png("Fig3C_LASSO_Path_SCI.png", width = 7, height = 5, units = "in", res = 300, family = "sans")

# 右边留宽一点给变量标签
par(mar = c(5, 5, 4, 6))

# 绘图
# lwd: 线条宽度
plot(cv_fit$glmnet.fit, xvar = "lambda", label = TRUE, lwd = 1.5, 
     xlab = "Log(Lambda)", ylab = "Coefficients", cex.lab = 1.2, cex.axis = 1.0)

# 辅助线
abline(v = log(cv_fit$lambda.min), lty = 2, lwd = 1.5, col = "black")
abline(v = log(cv_fit$lambda.1se), lty = 2, lwd = 1.5, col = "blue")

dev.off()
cat(">> Fig 3C Saved.\n")

# ==============================================================================
# Part C: 输出最终特征
# ==============================================================================
coef_1se <- coef(cv_fit, s = "lambda.1se")
selected_indices <- which(coef_1se != 0)
selected_indices <- selected_indices[selected_indices != 1] # 去掉Intercept
final_features <- rownames(coef_1se)[selected_indices]

cat("\n============================================\n")
cat("✅ Final Features for Python:\n")
cat(sprintf("['%s']", paste(final_features, collapse = "', '")), "\n")
cat("============================================\n")
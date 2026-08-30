import pandas as pd
import numpy as np

# ======================================================
# 0. 读取输入数据
# ======================================================
input_path  = r"D:\work\论文\Experiment\discriminant\relevance_heterogeneity.csv"     # ← 输入文件
output_path = r"D:\work\论文\Experiment\discriminant\discriminant_output.csv"    # ← 输出文件

df = pd.read_csv(input_path)

# ======================================================
# 1. 类型转换与保护
# ======================================================
df["idx"] = pd.to_numeric(df.get("idx"), errors="coerce")
df["m"]   = pd.to_numeric(df["m"], errors="coerce")
df["n"]   = pd.to_numeric(df["n"], errors="coerce")

if df["m"].isna().any() or df["n"].isna().any():
    print("⚠ 注意：存在无法转换为数值的 m 或 n（已设为 NaN），请检查原始数据。")

# ======================================================
# 2. 基于方差平衡的 m / n 标定（估计 beta）
# ======================================================
valid_mask = df["m"].notna() & df["n"].notna()

if valid_mask.sum() < 2:
    raise RuntimeError("有效的 (m, n) 样本数量不足，无法进行方差平衡标定。")

X = df.loc[valid_mask, "m"] ** 3
Y = df.loc[valid_mask, "n"] ** 2

sigma_X = X.std(ddof=1)
sigma_Y = Y.std(ddof=1)

if sigma_X <= 0 or sigma_Y <= 0 or np.isnan(sigma_X) or np.isnan(sigma_Y):
    beta = 1.0
    print("⚠ 警告：m^3 或 n^2 的标准差为 0 / NaN，beta 已退化为 1.0。")
else:
    beta = np.sqrt((8.0 * sigma_X) / (27.0 * sigma_Y))

print(f"👉 方差平衡得到 beta = {beta:.6f}")

# ======================================================
# 3. 映射为突变理论中的控制参量（新增列）
# ======================================================
df["m_std"] = -df["m"]
df["n_std"] =  df["n"] * beta

# ======================================================
# 4. 计算尖点突变判别式（新增列）
# ======================================================
df["discriminant"] = (
    8  * (df["m_std"] ** 3) +
    27 * (df["n_std"] ** 2)
)

# ======================================================
# 5. 导出：保留所有原始列 + 新增列
# ======================================================
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ 已完成：原始数据 + m_std / n_std / discriminant 已全部导出")
print(f"📄 输出文件路径：{output_path}")

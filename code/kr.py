import pandas as pd
import numpy as np

#====== Step 1 读取数据 ======#
file_path = r"D:\work\论文\Experiment\function\relevance_merged_with_heterogeneity.xlsx"   # 
df = pd.read_excel(file_path)

#====== Step 2 计算三类引用权重 ======#
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# 平均占比
F_d, F_m, F_l = df['P_d'].mean(), df['P_m'].mean(), df['P_l'].mean()

# 构建概率分布用于熵计算
q_d = df['P_d'] / df['P_d'].sum()
q_m = df['P_m'] / df['P_m'].sum()
q_l = df['P_l'] / df['P_l'].sum()

N = len(df)

# 分布集中度
C_d = 1 - entropy(q_d) / np.log(N)
C_m = 1 - entropy(q_m) / np.log(N)
C_l = 1 - entropy(q_l) / np.log(N)

# 综合贡献度 I_k
I_d = (1 - F_d + C_d) / 2
I_m = (1 - F_m + C_m) / 2
I_l = (1 - F_l + C_l) / 2

# 归一化得到权重
S = I_d + I_m + I_l
w_d, w_m, w_l = I_d / S, I_m / S, I_l / S

print("\n===== 计算得到三类引文权重 =====")
print(f"深度引用权重 w_d = {w_d:.4f}")
print(f"中度引用权重 w_m = {w_m:.4f}")
print(f"浅度引用权重 w_l = {w_l:.4f}\n")


#====== Step 3 计算每篇论文知识关联性 Relevance ======#
df['relevance'] = w_d * df['P_d'] + w_m * df['P_m'] + w_l * df['P_l']

#====== Step 4 导出结果 ======#
output_file = r"D:\work\论文\Experiment\function\relevance_heterogeneity.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"📌 已生成结果文件： {output_file}")
print("前5行结果预览：\n", df.head())

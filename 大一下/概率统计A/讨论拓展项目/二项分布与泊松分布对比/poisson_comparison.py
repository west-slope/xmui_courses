# -*- coding: utf-8 -*-
"""
泊松分布与二项分布对比图
在 VS Code 中可直接运行，会弹出图形窗口并保存图片文件。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

# 设置中文字体（如果系统无 SimHei 可注释掉，或改用英文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ------------------- 子图1：泊松近似较好的情况 -------------------
n1, p1 = 100, 0.1          # 二项分布参数
lambda1 = n1 * p1          # 泊松分布参数 λ = n * p
k1 = np.arange(0, 30)      # 取值范围（覆盖主要概率区域）

# 计算概率质量函数
binom_pmf1 = binom.pmf(k1, n1, p1)
poisson_pmf1 = poisson.pmf(k1, lambda1)

# 绘制条形图（并列显示）
width = 0.4
ax1.bar(k1 - width/2, binom_pmf1, width, alpha=0.7, 
        label=f'二项分布 (n={n1}, p={p1})', color='steelblue')
ax1.bar(k1 + width/2, poisson_pmf1, width, alpha=0.7, 
        label=f'泊松分布 (λ={lambda1:.1f})', color='darkorange')
ax1.set_xlabel('k (成功次数)')
ax1.set_ylabel('概率')
ax1.set_title('泊松近似较好：n=100, p=0.1')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# ------------------- 子图2：泊松近似较差的情况 -------------------
n2, p2 = 100, 0.5          # 二项分布参数
lambda2 = n2 * p2          # 泊松分布参数
k2 = np.arange(0, 80)      # 取值范围扩大

binom_pmf2 = binom.pmf(k2, n2, p2)
poisson_pmf2 = poisson.pmf(k2, lambda2)

ax2.bar(k2 - width/2, binom_pmf2, width, alpha=0.7, 
        label=f'二项分布 (n={n2}, p={p2})', color='steelblue')
ax2.bar(k2 + width/2, poisson_pmf2, width, alpha=0.7, 
        label=f'泊松分布 (λ={lambda2:.1f})', color='darkorange')
ax2.set_xlabel('k (成功次数)')
ax2.set_ylabel('概率')
ax2.set_title('泊松近似较差：n=100, p=0.5')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()

# 保存图片到当前目录
plt.savefig('poisson_binomial_comparison.png', dpi=300, bbox_inches='tight')
print("图片已保存为：poisson_binomial_comparison.png")

# 显示图形窗口
plt.show()
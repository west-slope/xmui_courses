import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, norm

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用于正常显示负号

# 创建子图布局：一行两个图（左：PDF，右：CDF）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==================== 参数设置 ====================
# 均匀分布参数
uniform_low = 0
uniform_high = 10
uniform_dist = uniform(loc=uniform_low, scale=uniform_high - uniform_low)

# 指数分布参数 (λ=0.5，即尺度参数=2)
expon_scale = 2.0  # 尺度参数 β = 1/λ
expon_dist = expon(scale=expon_scale)

# 正态分布参数 (μ=5, σ=2)
norm_mean = 5
norm_std = 2
norm_dist = norm(loc=norm_mean, scale=norm_std)

# ==================== 生成 x 轴数据 ====================
# 为每个分布单独定义x范围以展示完整形状
x_uniform = np.linspace(uniform_low - 1, uniform_high + 1, 1000)
x_expon = np.linspace(0, 12, 1000)      # 指数分布非负
x_norm = np.linspace(norm_mean - 3*norm_std, norm_mean + 3*norm_std, 1000)

# ==================== 计算PDF和CDF值 ====================
# 均匀分布
pdf_uniform = uniform_dist.pdf(x_uniform)
cdf_uniform = uniform_dist.cdf(x_uniform)

# 指数分布
pdf_expon = expon_dist.pdf(x_expon)
cdf_expon = expon_dist.cdf(x_expon)

# 正态分布
pdf_norm = norm_dist.pdf(x_norm)
cdf_norm = norm_dist.cdf(x_norm)

# ==================== 绘制概率密度函数图 (PDF) ====================
ax1.plot(x_uniform, pdf_uniform, 'b-', linewidth=2, 
         label=f'均匀分布 U({uniform_low},{uniform_high})')
ax1.plot(x_expon, pdf_expon, 'r-', linewidth=2, 
         label=f'指数分布 Exp(λ={1/expon_scale:.1f})')
ax1.plot(x_norm, pdf_norm, 'g-', linewidth=2, 
         label=f'正态分布 N(μ={norm_mean},σ={norm_std})')

ax1.fill_between(x_uniform, pdf_uniform, alpha=0.2, color='b')
ax1.fill_between(x_expon, pdf_expon, alpha=0.2, color='r')
ax1.fill_between(x_norm, pdf_norm, alpha=0.2, color='g')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('概率密度 f(x)', fontsize=12)
ax1.set_title('概率密度函数对比', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# ==================== 绘制累积分布函数图 (CDF) ====================
ax2.plot(x_uniform, cdf_uniform, 'b-', linewidth=2, 
         label=f'均匀分布 U({uniform_low},{uniform_high})')
ax2.plot(x_expon, cdf_expon, 'r-', linewidth=2, 
         label=f'指数分布 Exp(λ={1/expon_scale:.1f})')
ax2.plot(x_norm, cdf_norm, 'g-', linewidth=2, 
         label=f'正态分布 N(μ={norm_mean},σ={norm_std})')

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('累积概率 F(x)', fontsize=12)
ax2.set_title('累积分布函数对比', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 添加参考线（y=0.5 中位数参考线）
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(ax2.get_xlim()[1]*0.98, 0.5, '中位数', ha='right', va='bottom', 
         fontsize=9, color='gray')

plt.tight_layout()
plt.show()

# ==================== 补充信息输出 ====================
print("="*50)
print("分布参数信息：")
print(f"均匀分布: 区间 [{uniform_low}, {uniform_high}]")
print(f"指数分布: 速率 λ = {1/expon_scale:.2f}, 均值 = {expon_dist.mean():.2f}")
print(f"正态分布: 均值 μ = {norm_mean}, 标准差 σ = {norm_std}")
print("="*50)

# 可选：绘制三维对比图（拓展创意部分）
fig2, ax3 = plt.subplots(figsize=(10, 4))
x_common = np.linspace(-2, 12, 1000)
pdf_u_common = uniform_dist.pdf(x_common)
pdf_e_common = expon_dist.pdf(x_common)
pdf_n_common = norm_dist.pdf(x_common)

ax3.stackplot(x_common, pdf_u_common, pdf_e_common, pdf_n_common, 
              labels=['均匀分布', '指数分布', '正态分布'],
              colors=['blue', 'red', 'green'], alpha=0.6)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('概率密度', fontsize=12)
ax3.set_title('堆叠面积图：三种分布概率密度的叠加对比', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
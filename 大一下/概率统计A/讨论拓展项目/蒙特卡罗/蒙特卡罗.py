import random
import math

def f(x):
    """被积函数"""
    return x**2

def monte_carlo_drop_points(N, a=0, b=1):
    """
    方法1：随机投点法
    在矩形 [a,b] × [0, M] 内随机投点，统计曲线下方点数
    """
    M = 1  # f(x)=x²在[0,1]上的最大值为1
    
    points_under_curve = 0
    
    for _ in range(N):
        # 随机投点
        x = random.uniform(a, b)
        y = random.uniform(0, M)
        
        # 判断是否在曲线下方
        if y <= f(x):
            points_under_curve += 1
    
    # 面积 = 矩形面积 × (下方点数/总点数)
    rectangle_area = M * (b - a)
    integral = rectangle_area * (points_under_curve / N)
    
    return integral

# 测试
N = 100000
result = monte_carlo_drop_points(N)
exact = 1/3  # ∫₀¹ x² dx = 1/3

print(f"随机投点法结果: {result:.6f}")
print(f"精确值: {exact:.6f}")
print(f"误差: {abs(result - exact):.6f}")
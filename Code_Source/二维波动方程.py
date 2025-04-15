# mathmodel13_v1.py
# Demo10 of mathematical modeling algorithm
# Solving partial differential equations
# 偏微分方程数值解法

# 5. 二维热传导方程（抛物型二阶偏微分方程）
# pu/p2 = c*(p2u/px2+p2u/py2)

import numpy as np
import matplotlib.pyplot as plt

def showcontourf(zMat, xyRange, tNow):  # 绘制等温云图
    x = np.linspace(xyRange[0], xyRange[1], zMat.shape[1])
    y = np.linspace(xyRange[2], xyRange[3], zMat.shape[0])
    xx,yy = np.meshgrid(x,y)
    zMax = np.max(zMat)
    yMax, xMax = np.where(zMat==zMax)[0][0], np.where(zMat==zMax)[1][0]
    levels = np.arange(0,100,1)
    showText = "time = {:.1f} s\nhotpoint = {:.1f} C".format(tNow, zMax)

    plt.clf()  # 清除当前图形及其所有轴，但保持窗口打开
    plt.plot(x[xMax],y[yMax],'ro')  # 绘制最高温度点
    plt.contourf(xx, yy, zMat, 100, cmap=plt.cm.get_cmap('jet'), origin='lower', levels = levels)
    plt.annotate(showText, xy=(x[xMax],y[yMax]), xytext=(x[xMax],y[yMax]),fontsize=10)
    plt.colorbar()
    plt.xlabel('Xupt')
    plt.ylabel('Youcans')
    plt.title('Temperature distribution of the plate')
    plt.pause(0.01)
    # plt.show()

# 模型参数
uIni = 25  # 初始温度值
uBound = 25.0  # 边界条件
c = 1.0  # 热传导参数
qv = 50.0  # 热源功率
x0, y0 = 0.0, 3.0  # 热源初始位置
vx, vy = 2.0, 1.0  # 热源移动速度
# 求解范围
tc, te = 0.0, 5.0  # 时间范围，0<t<te
xa, xb = 0.0, 16.0  # 空间范围，xa<x<xb
ya, yb = 0.0, 12.0  # 空间范围，ya<y<yb

# 初始化
dt = 0.002  # 时间步长
dx = dy = 0.1  # 空间步长
tNodes = round(te/dt)  # t轴 时序网格数
xNodes = round((xb-xa)/dx)  # x轴 空间网格数
yNodes = round((yb-ya)/dy)  # y轴 空间网格数
xyRange = np.array([xa, xb, ya, yb])
xZone = np.linspace(xa, xb, xNodes+1)  # 建立空间网格
yZone = np.linspace(ya, yb, yNodes+1)  # 建立空间网格
xx,yy = np.meshgrid(xZone, yZone)  # 生成网格点的坐标 xx,yy (二维数组)

# 计算 差分系数矩阵 A、B (三对角对称矩阵)，差分系数 rx,ry,ft
A = (-2) * np.eye(xNodes+1, k=0) + (1) * np.eye(xNodes+1, k=-1) + (1) * np.eye(xNodes+1, k=1)
B = (-2) * np.eye(yNodes+1, k=0) + (1) * np.eye(yNodes+1, k=-1) + (1) * np.eye(yNodes+1, k=1)
rx, ry, ft = c*dt/(dx*dx), c*dt/(dy*dy), qv*dt

# 计算 初始值
U = uIni * np.ones((yNodes+1, xNodes+1))  # 初始温度 u0

# 前向欧拉法一阶差分求解
for k in range(tNodes+1):
    t = k * dt  # 当前时间

    # 热源条件
    # (1) 恒定热源：Qv(x0,y0,t) = qv, 功率、位置 恒定
    # Qv = qv
    # (2) 热源功率随时间变化 Qv(x0,y0,t)=f(t)
    # Qv = qv*np.sin(t*np.pi) if t<2.0 else qv
    # (3) 热源位置随时间变化 Qv(x,y,t)=f(x(t),y(t))
    xt, yt = x0+vx*t, y0+vy*t  # 热源位置变化
    Qv = qv * np.exp(-((xx-xt)**2+(yy-yt)**2))  # 热源方程

    # 边界条件
    U[:,0] = U[:,-1] = uBound
    U[0,:] = U[-1,:] = uBound

    # 差分求解
    U = U + rx * np.dot(U,A) + ry * np.dot(B,U) + Qv*dt

    if k % 100 == 0:
        showcontourf(U, xyRange, k*dt)  # 绘制等温云图
        print('t={:.2f}s\tTmax={:.1f}  Tmin={:.1f}'.format(t, np.max(U), np.min(U)))
plt.show()

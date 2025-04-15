import matplotlib as mpl
import matplotlib.pyplot as plt  # 绘图工具库
import numpy as np  # 科学计算库 处理多维数据(矩阵)

# 设置基本参数
global Row, Col  # 定义全局变量行和列
Row = 100  # 行
Col = 100  # 列
forest_area = 0.8  # 初始化这个地方是树木的概率
firing = 0.8  # 绿树受到周围树木引燃的概率
grow = 0.001  # 空地上生长出树木的概率
lightning = 0.00006  # 闪电引燃绿树的概率
forest = (np.random.rand(Row, Col) < forest_area).astype(np.int8)
# 初始化作图
plt.title("step=1", fontdict={"family": 'Times New Roman', "weight": 'bold', "fontsize": 20})  # 字体，加粗，字号
colors = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]  # 黑色空地 绿色树 红色火
bounds = [0, 1, 2, 3]  # 类数组，单调递增的边界序列
cmap = mpl.colors.ListedColormap(colors)  # 从颜色列表生成颜色的映射对象
w = plt.imshow(forest, cmap=cmap, norm=mpl.colors.BoundaryNorm(bounds, cmap.N))

# 迭代
T = 500  # 迭代500次
for t in range(T):
    temp = forest  # 上一个状态的森林
    temp = np.where(forest == 2, 0, temp)  # 燃烧的树变成空地
    p0 = np.random.rand(Row, Col)  # 空位变成树木的概率
    temp = np.where((forest == 0) * (p0 < grow), 1, temp)  # 如果这个地方是空位，满足长成绿树的条件，那就变成绿树
    fire = (forest == 2).astype(np.int8)  # 找到燃烧的树木
    firepad = np.pad(fire, (1, 1), 'wrap')  # 上下边界，左右边界相连接
    numfire = firepad[0:-2, 1:-1] + firepad[2:, 1:-1] + firepad[1:-1, 0:-2] + firepad[1:-1, 2:]
    p21 = np.random.rand(Row, Col)  # 绿树因为引燃而变成燃烧的树
    p22 = np.random.rand(Row, Col)  # 绿树因为闪电而变成燃烧的树
    # Temp = np.where((forest == 1)&(((numfire>0)&(rule1prob<firing))|((numfire==0)&(rule3prob<lightning))),2,Temp)
    temp = np.where((forest == 1) & (((numfire > 0) & (p21 < firing)) | ((numfire == 0) & (p22 < lightning))), 2, temp)

    forest = temp  # 更新森林的状态
    plt.title("step=" + str(t), fontdict={"family": 'Times New Roman', "weight": 'bold', "fontsize": 20})  # 字体，加粗，字号
    w.set_data(forest)
    plt.savefig('./元胞自动机//' + str(t) + '.jpg')
    plt.pause(0.1)
# plt.show()
from PIL import Image

im = Image.open("./元胞自动机//0.jpg")
images = []
for i in range(1, 200):
    images.append(Image.open('./元胞自动机//' + str(i) + '.jpg'))
im.save('1125.gif', save_all=True, append_images=images, duration=100)

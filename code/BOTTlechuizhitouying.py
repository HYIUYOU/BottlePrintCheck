# 导入所需模块
import cv2
from matplotlib import pyplot as plt


# 显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# plt显示彩色图片
def plt_showA(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# %%

# 加载图片
rawImage = cv2.imread("/Users/heyiyuan/Desktop/BottlePrintCheck/pics/test1.png")
plt_showA(rawImage)

# %%

# 灰度处理
image = rawImage.copy()
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt_show(gray_image)

# %% md

# 字符水平方向的切割
## 目的：去除车牌边框和铆钉的干扰

# %%

# 自适应阈值处理(二值化)
ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
plt_show(image)

# %%

image.shape  # 47行，170列
rows = image.shape[0]
cols = image.shape[1]
print(rows, cols)

# %%

# 二值统计,统计没每一行的黑值（0）的个数
hd = []
for row in range(rows):
    res = 0
    for col in range(cols):
        if image[row][col] == 0:
            res = res + 1
    hd.append(res)
len(hd)
max(hd)

# %%

# 画出柱状图
y = [y for y in range(rows)]
x = hd
plt.barh(y, x, color='black', height=1)
# 设置x，y轴标签
plt.xlabel('0_number')
plt.ylabel('row')
# 设置刻度
plt.xticks([x for x in range(0, 130, 5)])
plt.yticks([y for y in range(0, rows, 1)])

plt.show()

# %% md

### 中间较为密集的地方就是车牌有字符的地方,从而很好的去除了牌边框及铆钉
#从图中可以明显看出车牌字符区域的投影值和车牌边框及铆钉区域的投影值之间明显有一个波谷，找到此处波谷，就可以得到车牌的字符区域，去除车牌边框及铆钉。

# %%

x = range(int(rows / 2), 2, -1)
x = [*x]
x

# %%

# 定义一个算法,找到波谷,定位车牌字符的行数区域
# 我的思路;对于一个车牌,中间位置肯定是有均匀的黑色点的,所以我将图片垂直分为两部分,找波谷
mean = sum(hd[0:int(rows / 2)]) / (int(rows / 2) + 1)
mean
region = []
for i in range(int(rows / 2), 2, -1):  # 0,1行肯定是边框,直接不考虑,直接从第二行开始
    if hd[i] < mean:
        region.append(i)
        break
for i in range(int(rows / 2), rows):  # 0,1行肯定是边框,直接不考虑,直接从第二行开始
    if hd[i] < mean:
        region.append(i)
        break
region

# %%

image1 = image[region[0]:region[1], :]  # 使用行区间

# %%

plt_show(image1)

# %% md

# 字符垂直方向的切割

# %%

image11 = image1.copy()

# %%

image11.shape  # 47行，170列
rows = image11.shape[0]
cols = image11.shape[1]
print(rows, cols)

# %%

cols  # 170列

# %%

# 二值统计,统计没每一列的黑值（0）的个数
hd1 = []
for col in range(cols):
    res = 0
    for row in range(rows):
        if image11[row][col] == 0:
            res = res + 1
    hd1.append(res)
len(hd1)
max(hd1)

# %%

# 画出柱状图
y = hd1  # 点个数
x = [x for x in range(cols)]  # 列数
plt.bar(x, y, color='black', width=1)
# 设置x，y轴标签
plt.xlabel('col')
plt.ylabel('0_number')
# 设置刻度
plt.xticks([x for x in range(0, cols, 10)])
plt.yticks([y for y in range(0, max(hd1) + 5, 5)])

plt.show()

# %%

mean = sum(hd1) / len(hd1)
mean

# %%

# 简单的筛选
for i in range(cols):
    if hd1[i] < mean / 4:
        hd1[i] = 0

# %%

# 画出柱状图
y = hd1  # 点个数
x = [x for x in range(cols)]  # 列数
plt.bar(x, y, color='black', width=1)
# 设置x，y轴标签
plt.xlabel('col')
plt.ylabel('0_number')
# 设置刻度
plt.xticks([x for x in range(0, cols, 10)])
plt.yticks([y for y in range(0, max(hd1) + 5, 5)])

plt.show()

# %% md

## 从直方图中可以看到很多波谷,这些就是字符分割区域的黑色点的个数等于0,我们就可以通过这些0点进行分割,过滤掉这些不需要的部分部分

# %%

# 找所有不为0的区间(列数)
region1 = []
reg = []
for i in range(cols - 1):
    if hd1[i] == 0 and hd1[i + 1] != 0:
        reg.append(i)
    if hd1[i] != 0 and hd1[i + 1] == 0:
        reg.append(i + 2)
    if len(reg) == 2:
        if (reg[1] - reg[0]) > 5:  # 限定区间长度要大于5(可以更大),过滤掉不需要的点
            region1.append(reg)
            reg = []
        else:
            reg = []
region1

# %%

# 测试
image2 = image1[:, region1[0][0]:region1[0][1]]
plt_show(image2)

# %%

# 为了使字符之间还是存在空格,定义一个2像素白色的区域
import numpy as np

white = []
for i in range(rows * 2):
    white.append(255)
white = np.array(white)
white = white.reshape(rows, 2)
white.shape

# %%

#  遍历所有区域,保存字符图片到列表
p = []
for r in region1:
    r = image1[:, r[0]:r[1]]
    plt_show(r)
    p.append(r)
    p.append(white)

# %%

# 将字符图片列表拼接为一张图
image2 = np.hstack(p)

# %%

plt_show(image2)

# %%

# 将分割好的字符图片保存到文件夹
print(region)
print(region1)

# %%

plt_showA(rawImage)

# %%

v_image = rawImage[region[0]:region[1], :]
plt_showA(v_image)

# %%

i = 1
for reg in region1:
    h_image = v_image[:, reg[0]:reg[1]]
    plt_showA(h_image)
    cv2.imwrite('./words/test4_' + str(i) + '.png', h_image)
    i = i + 1

# %%

word_images = []
for i in range(1, 8):
    word = cv2.imread('./words/test4_' + str(i) + '.png', 0)
    ret, word = cv2.threshold(word, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    word_images.append(word)
word_images
plt.imshow(word_images[0], cmap='gray')
for i, j in enumerate(word_images):
    plt.subplot(1, 8, i + 1)
    plt.imshow(word_images[i], cmap='gray')
plt.show()

# %%


# %%



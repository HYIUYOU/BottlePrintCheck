import cv2
import numpy as np
import time
import math
from multiprocessing import Process, Value

arr1=[]
arr2=[]

#像素抖动误差 x为水平方向 y为垂直方向
x=30
y=25

result = True

# 使用openCV的预处理方法
def pretreatWithOpenCV(img_path):
    img = cv2.imread(img_path)
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bw_image = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
    return np.array(bw_image)


# 预处理方法
def pretreat(ima):

    a=time.time()
    ima=ima.convert('L')         #转化为灰度图像
    im=np.array(ima)

    b=time.time()
    print("读图和灰度化耗时:", b - a)

    row=im.shape[0]
    col=im.shape[1]
    #转化为二维数组
    for i in range(row):#转化为二值矩阵
        for j in range(col):
            #去除上下干扰
            if (i<100 or i>row-150):
                im[i][j]=0
                continue

            if im[i,j]>=100:#以灰度100为分界
                im[i,j]=255
            else:
                im[i,j]=0

    c=time.time()
    print("二值化耗时:",c-b)
    return im


# 获取中心线
def getMid(img):
    cols = img.shape[1]
    up = 200
    down = 600

    first = -1
    last = 0
    mid = 0
    mids = []

    for i in range(up, down):  # 遍历每行
        for j in range(cols):
            if (img[i][j] == 255):
                if(first==-1):
                    first = j
                    last = j
                else:
                    last=j
            else:
                continue
        if (first * last >= 0 and last - first > 600):
            mids.append((first + last) / 2)

        first = -1
        last = 0

    if(len(mids)<=0):
        return -1

    for k in mids:
        mid=mid+k
    mid= mid / len(mids)

    return math.ceil(mid)

# 平移
def translate(img,bias):
    rows = img.shape[0]
    cols = img.shape[1]
    tmpline=[]

    for i in range(rows):
        tmpline=img[i].copy()
        for j in range(cols):
            if(j+bias>=0 and j+bias<cols):
                img[i][j + bias] = tmpline[j]
            else:
                continue

        if(bias>=0):
            for k in range(bias):
                img[i][k]=0
        else:
            for k in range(bias):
                img[i][cols-k]=0

    return img

'''
整块图片检测方法  参数：原始图像，比较图像
'''
def test(im1,im2):

    rows=im1.shape[0]
    cols=im1.shape[1]

    count=0
    for i in range(rows):  #遍历每个像素点
        for j in range(cols):
            count = count + 1
            if(im1[i][j]==0):#二值图像0为背景，忽略；只匹配原始图像值为255的像素点
                continue
            flag=0

            l=j-x#左边界
            if (l < 0):
                l = 0

            u=i-y#上
            if(u<0):
                u=0

            r=j+x#右
            if(r>=cols):
                r=cols

            d=i+y#下
            if(d>=rows):
                d=rows

            # print(l,r,u,d)

            for jj in range(l,r):#开始匹配
                for ii in range(u,d):
                    if(im1[i][j]==im2[ii][jj]):
                        flag=1
                        break
                if(flag==1):
                    break

            if(flag==0):
                print("遍历像素个数:",count)
                print("出错位置:",i,j)
                print("行&列像素数:",rows, cols)
                return False
    print("遍历像素个数:",count)
    print("行&列像素数:",rows,cols)
    return True


''' 
多进程检测方法
im1 - 原图片，im2 - 待测图片，
blockid - 区块号（逐行记号  e.g. 4块则0/1/2/3分别对应左上/右上/左下/右下)
slice - 图片分块方式 (x,y)- x行y列
no_fault - 用于传递结果
'''
def multiProcess_test(im1,im2,blockid, slice, no_fault):


    rows_begin = im1.shape[0] * (blockid//slice[1]) // slice[0]
    rows_end =  im1.shape[0] * (blockid//slice[1] + 1) // slice[0]

    cols_begin = im1.shape[1] * (blockid%slice[1]) // slice[1]
    cols_end = im1.shape[1] * (blockid%slice[1] + 1) // slice[1]

    count=0
    for i in range(rows_begin, rows_end):  # 遍历每个像素点
        for j in range(cols_begin, cols_end):
            count = count + 1
            if(im1[i][j]==0): # 二值图像0为背景，忽略；只匹配原始图像值为1的像素点
                continue

            flag=0

            l=j-x#左边界
            if (l < 0):
                l = 0
            u=i-y#上
            if(u<0):
                u=0
            r=j+x#右
            if(r>=cols_end):
                r=cols_end
            d=i+y#下
            if(d>=rows_end):
                d=rows_end

            for jj in range(l, r):  # 开始匹配
                for ii in range(u, d):
                    if (im1[i][j]==im2[ii][jj]):
                        flag = 1
                        break
                if (flag == 1):
                    break

            if(flag==0):
                print("------")
                print("Block ", blockid)
                print("遍历像素个数:",count)
                print("出错位置:",i,j)
                print("行&列像素数:", rows_end - rows_begin, cols_end - cols_begin)
                print("匹配结果:",False)
                no_fault.value = 0
                return
    print("------")
    print("Block ", blockid)
    print("遍历像素个数:",count)
    print("行&列像素数:",rows_end-rows_begin,cols_end-cols_begin)
    print("匹配结果",True)
    # print("rows begin:",rows_begin," ,end:",rows_end)
    # print("cols begin:", cols_begin, " ,end:", cols_end)

    return

if __name__ == '__main__':

    start = time.time()
    start_load = time.time()

    img1 = pretreatWithOpenCV('../pics/A1-1.bmp') # 原始图像路径
    img2 = pretreatWithOpenCV('../pics/A1-1.bmp') # 待测图像路径
    # img2 = pretreatWithOpenCV('../pics/A1-1-3L.bmp') # 有缺印图像
    # img2 = pretreatWithOpenCV('../pics/B1-1.bmp') # 有缺印图像


    # 去除瓶身上下亮度干扰
    img_row=img2.shape[0]
    img_col=img2.shape[1]
    for i in range(0,100):
        for j in range(img_col):
            img1[i][j] = 0
            img2[i][j] = 0

    for i in range(img_row-150, img_row):
        for j in range(img_col):
            img1[i][j] = 0
            img2[i][j] = 0


    end_load = time.time()
    print("读入并预处理图片时间：", end_load - start_load)


    start_trans = time.time()
    mid = getMid(img2)
    print("中心线位置:", mid)
    img = translate(img2, 618 - mid)
    end_trans = time.time()
    print("获取中心线并进行平移时间：", end_trans - start_trans)

    start_match = time.time()

    no_fault = Value('I',1)

    processes = []

    '''
    图片分块方式 
    (x,y)- 分成x行y列
    '''
    slice = (4,2)

    pNum = slice[0]*slice[1]  # 计算并发的进程数
    # 创建进程并开始检测
    for pid in range(0, pNum):
        processes.append(Process(target=multiProcess_test, args=(img1, img2, pid, slice, no_fault)))

    for process in processes:
         process.start()

    count=processes.__len__()
    while(True):
        for curr_process in processes:
            if(curr_process.is_alive()):
                pass
            else:
                count = count-1

        # 已检测出错误，停止所有子块的匹配
        if (no_fault.value == 0):
            for curr_process in processes:
                if (curr_process.is_alive()):
                    curr_process.terminate()
                else:
                    continue
            break

        if(count==0):
            break
        count=processes.__len__()

    result_str = "无缺印"
    if no_fault.value != 1:
        result_str = "有缺印"

    end=time.time()
    print("------")
    print("结果：",result_str)
    print("匹配时间",end-start_match)
    print("运行时间:",end-start)



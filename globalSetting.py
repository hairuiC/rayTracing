import math

from math import *
import numpy as np
# from numpy.core._multiarray_umath import sign
# from numpy.core._multiarray_umath import dot

PI = pi
vec = np.array
sqrt = math.sqrt
sign = np.sign
dot = np.dot
cross = np.cross
exp = np.exp
norm = np.linalg.norm
TMIN = 0.001 # 移动一小步避免自相交
TMAX = 2000000.0 # 最大单词光线传播距离
Max_Step = 512 # 最大光线步进次数
Max_Trace = 512
Judge_hit = 0.0001
ENV_IOR = 1.000277

image_plane = 0

RECEIVER_WIDTH = 320
RECEIVER_HEIGHT= 320
RECEIVER = np.zeros((RECEIVER_WIDTH, RECEIVER_HEIGHT))
# WIDTH_dis = normal_discrete
count_mirror=0
count_ref=0
count_receiver=0
ini_vec = vec([0, 0, 0])
DIS_LVL = 10
light_pattern = np.zeros((4, 4))

att_kc = 1.0
att_kl = 0.045
att_kq = 0.0075

def dot2(v):
    return np.dot(v, v)

def clamp(x, xmin, xmax):
    # print(max(xmin, x))
    # print(np.min(xmax, np.max(xmin, x)))
    return min(xmax, max(xmin, x))

import random

import numpy as np
import math
from globalSetting import *
from dataStructure import *
import matplotlib.pyplot as plt
import cv2
def random_in_unit_disk(r):  # 单位圆内随机取一点
    x = np.random.uniform(0, r)
    # a = np.random.random() * 2 * PI
    a = np.random.random() * 2 * PI

    return sqrt(x) * cos(a), sqrt(x) * sin(a)

def cal_norm(a, b, c):
    x1, y1, z1 = a[0], a[1], a[2]
    x2, y2, z2 = b[0], b[1], b[2]
    x3, y3, z3 = c[0], c[1], c[2]

    A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
    C = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
    B = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

    norm = vec([A, B, C]) / np.linalg.norm(vec(A, B, C))
    return norm

def zebra_light_origin(step):
    # linspace = np.linspace(0, 1, 6)
    # print(linspace)
    light_ori = []
    x = 0.3
    while(int(x / step) % 2 != 0):
        x = np.random.uniform(0, 1)
        # if int(x / step) % 2 != 0:
        #     continue
    z = random.uniform(0, 1)
    return x, z

def attenuation(distance):
    return 1.0 / (att_kc + att_kl * distance + att_kq * (distance * distance))


        # print(x, z)

def TBN(N: vec): # 用世界坐标下的法线计算 TBN 矩阵
    T = vec([0, 0, 0])
    B = vec([0, 0, 0])

    if N[1] < -0.99999:
        T = vec([0, -1, 0])
        B = vec([-1, 0, 0])
    else:
        a = 1.0 / (1.0 + N[2]) # float
        b = -N[0] * N[1] * a

        T = vec([1.0 - N[0] * N[0] * a, b, -N[0]])
        B = vec([b, 1.0 - N[1] * N[1] * a, -N[1]])

    return np.matrix([T, B, N])


def rotation_matrix_from_vectors(source_vec, target_vec):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (source_vec / np.linalg.norm(source_vec)).reshape(3), (target_vec / np.linalg.norm(target_vec)).reshape(3)
    print(a, b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix



    # for i in range(int(1 / step)):
    #     if i % 2 == 0:
    #         x = random.uniform()
    #
    #
    #
    #
    # x = random.uniform()
def unit_function(vec):
    return vec / np.linalg.norm(vec)


def hemispheric_sampling(n):  # 以 n 为法线进行半球采样
    u, v = np.random.rand(2)
    phi = 2 * np.pi * u
    theta = np.arccos(1 - v)
    x = sqrt(v) * np.cos(phi)
    z = sqrt(v) * np.sin(phi)
    y = sqrt(1 - v)
    # print(TBN(n))

    direction = TBN(n) @ vec([x, y, z])
    return direction
    # ra = np.random.rand * 2 * pi
    # rb = ti.random()

    # rz = sqrt(rb)
    # v = vec2(cos(ra), sin(ra))
    # rxy = sqrt(1.0 - rb) * v
    #
    # return TBN(n) @ vec3(rxy, rz)

def get_ray(lightSourceType, middle, radius, Pad_norm, step=0.):
    if lightSourceType == 'parallel':
        x_coor, z_coor = random_in_unit_disk(radius)
        # print(vec([0, 0, 0]) - middle)
        # print(np.linalg.norm(vec([0, 0, 0]) - middle))
        direction = (vec([0, 0, 0]) - middle)/np.linalg.norm(vec([0, 0, 0]) - middle)
        height = middle[1]
        ray_origin = vec([x_coor + 6.25, height, z_coor])
        return Ray(origin=ray_origin, direction=direction)
    if lightSourceType == 'zebra':
        height = middle[1]

        light_a = vec([middle[0] + radius,  height, middle[2] + radius])
        light_b = vec([middle[0] - radius,  height, middle[2] + radius])
        light_c = vec([middle[0] - radius,  height, middle[2] - radius])
        light_d = vec([middle[0] + radius,  height, middle[2] - radius])
        direction = (vec([0, 0, 0]) - middle) / np.linalg.norm(vec([0, 0, 0]) - middle)

        x_ratio, z_ratio = zebra_light_origin(step)
        x = (x_ratio*radius*2) + min(light_a[0], light_b[0], light_c[0], light_d[0])
        z = (z_ratio*radius*2) + min(light_a[2], light_b[2], light_c[2], light_d[2])

        ray_origin = vec([x, height, z])
        return Ray(origin=ray_origin, direction=direction)
    if lightSourceType == 'lightPad':
        # LCD板子上每一个点都是一个点光源
        ray_list = []
        # rota = rotation_matrix_from_vectors(vec([0, 1, 0]), Pad_norm)
        # mat_TBN = TBN(Pad_norm)
        for i in range(10000): #每个点光源发100根光线
            direction = hemispheric_sampling(Pad_norm)
            # print(np.array(direction)[0])
            # mat_TBN = TBN()

            # u, v = np.random.rand(2)
            # phi = 2 * np.pi * u
            # theta = np.arccos(1 - v)
            # x = sqrt(v) * np.cos(phi)
            # z = sqrt(v) * np.sin(phi)
            # y = sqrt(1 - v)
            # # print(np.linalg.norm(vec([x, y, z])))
            #
            # rota_dir = rota* vec([x, y, z])
            # rota_dir = rota_dir[:, 1]
            # direction = vec(rota_dir)
            # print(np.linalg.norm(direction))
            # direction = direction / np.linalg.norm(direction)
            # print(np.linalg.norm(direction))

            # print(rota_dir)
            # print(np.linalg.norm(vec(rota_dir)))
            # print(rota_dir / np.linalg.norm(vec(rota_dir)))
            # direction = vec(rota_dir) / np.linalg.norm(vec(rota_dir))
            ray_origin = vec((middle[0], middle[1], middle[2]))
            direction = np.array(direction)[0]
            ray_list.append(Ray(origin=ray_origin, direction=vec(direction)))
        return ray_list


        # return
        # print(x, z)


        # return


        # light_b = vec()
# for i in range(1000):
#     get_ray('zebra', middle=vec([6.25, 6.25, 0]), radius=1, step=0.2)






def RussianRoulette():
    light_quality = 1 / 50
    inv_pdf = exp(float(512) * light_quality)
    roulette_prob = 1.0 - (1.0 / inv_pdf)
    return roulette_prob

def pow5(x): # 快速计算 x 的 5 次方
    t = x*x
    t *= t
    return t*x


def fresnel_schlick(cosine, F0):
    return F0 + (1 - F0) * pow5(abs(1 - cosine))

def fresnel_schlick_roughness(cosine: float, F0: float, roughness: float) -> float:  # 计算粗糙度下的菲涅尔近似值
    return F0 + (max(1 - roughness, F0) - F0) * pow5(abs(1 - cosine))

def reflect(inDir, normal):
    k = np.dot(inDir, normal)
    return inDir - 2.0 * k * normal

def normal_discrete(plane:plane, hit_pos, ray):
    # np.zeros((RECEIVER_WIDTH*dis_level, RECEIVER_HEIGHT*dis_level))
    # x_ratio = hit_pos[0] / abs(plane.b[0] - plane.a[0])
    # z_ratio = hit_pos[2] / abs(plane.c[2] - plane.b[2])
    #
    #
    x = hit_pos[0]
    y = hit_pos[2]
    # print(hit_pos)
    if (x < min(plane.a[0], plane.b[0], plane.c[0], plane.d[0]) or x > max(plane.a[0], plane.b[0], plane.c[0], plane.d[0])) or (y < min(plane.a[2], plane.b[2], plane.c[2], plane.d[2]) or y > max(plane.a[2], plane.b[2], plane.c[2], plane.d[2])):
        return
    # print(min(plane.a[0], plane.b[0], plane.c[0], plane.d[0]))
    x_ratio = (x - min(plane.a[0], plane.b[0], plane.c[0], plane.d[0])) / RECEIVER_WIDTH
    y_ratio = (y - min(plane.a[2], plane.b[2], plane.c[2], plane.d[2])) / RECEIVER_HEIGHT
    # print(x_ratio*RECEIVER_WIDTH)
    RECEIVER[int(x_ratio*RECEIVER_WIDTH*DIS_LVL), int(y_ratio*RECEIVER_HEIGHT*DIS_LVL)] += ray.Radiance
    # dis = np.linspace(0, 1, dis_level) # x, y, z都要
    # dis_xup, dis_yup, dis_zup = 0, 0, 0
    # dis_xlow, dis_ylow, dis_zlow = 0, 0, 0
    #
    # for i in range(dis_level-1):
    #     if min(plane.a[0], plane.b[0]) + (abs(plane.a[0] - plane.b[0])*dis[i+1]) > hit_pos[0]:
    #         dis_xup = dis[i+1]
    #         dis_xlow = dis[i]
    #         break
    #
    # for k in range(dis_level-1):
    #     if min(plane.a[2], plane.b[2]) + (abs(plane.a[2] - plane.b[2])*dis[k+1]) > hit_pos[2]:
    #         dis_zup = dis[k+1]
    #         dis_zlow = dis[k]
    #         break
    # return dis_xup, dis_zup, dis_xlow, dis_zlow

def check_close(ray, normal):
    if np.sign(np.dot(ray.Direction, normal) * np.dot(ray.Ray_Origin, normal)) < 0:
        return True
    else:
        return False

def visualization(img):
    plt.imshow(img, cmap='hot')
    plt.colorbar()
    plt.show()
    plt.savefig('roughness_0_circle.jpg')

def mapping23(img_2D, plane_3D:plane):
    emission = np.where(np.matrix(img_2D))
    light = []
    for emi in emission:
        i = emi[0]
        j = emi[1]
        x = i + plane_3D.a[0]
        y = j + plane_3D.a[1]
        z = (-(plane_3D.D + plane_3D.A * x + plane_3D.B * y)/plane_3D.C)
        temp_emi = vec([x, y, z])
        light.append(temp_emi)
    return light

# def intersect_plane(ray, plane):
#     pass
#
#
#
# def hit_camera(cam:Camera, ray:Ray):
#     # 先确定击中camera，再确定击中image plane
#     vec1 = cam.pos - ray.Ray_Origin
#     # vec2 = ray.Direction
#     if np.dot(ray.Direction, cam.lookat) > 0:
#         # 和视线同向，从相机背面击中相机
#         return False
#     vec1 = unit_function(vec1)
#     cos_theta = np.dot(vec1, ray.Direction) / (np.linalg.norm(vec1) * np.linalg.norm(ray.Direction))
#     if np.abs(cos_theta - 1) == 0:
#         return True
#     else:
#         return False



        # 击中相机，接下来需要击中平面




# def check_onboard(hit_pos, plane:plane):



# a = vec([0, -1, 0])
# b = vec([-0.44, 0.00009, -0.56])
# print(np.sign(np.dot(a, b)))
    # dis_y = np.linspace(0, 1, dis_level)
    # dis_z = np.linspace(0, 1, dis_level)


# a = vec([0, 1, 0])
# b = a
# b *= -1
# print(a)




    # dis_x = np.linspace(0, 1, dis_level)*RECEIVER_WIDTH
    # dis_z = np.linspace(0, 1, dis_level)*RECEIVER_HEIGHT


    # for i in range(len(dis_x)):
    #     if coor[0] < dis_x[i]:
    #         upx = dis_x[i]
    #         lowx = dis_x[i-1]
    #         upz = dis_z[i]
    #         lowz = dis_z[i-1]
    #         break
    # return upx, lowx, upz, lowz
# dis = np.linspace(0, 1, 100)
# print(dis)


    # dis_z = np.




    # for i in range(len(dis)):
    #     if coor < dis[i]:
    #         upx = dis[i]
    #         lowx = dis[i-1]
    #         return upx, lowx


# normal_discrete()
#
#
#
#
# a = vec([1, 2, 3])
# n = vec([0, 1, 0])
#
# print(reflect(a, n))


# a = np.random.normal(1, 0, 10)
# print(a)
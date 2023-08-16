import numpy as np
from LightSourceCustom import *
from dataStructure import *
# from MirrorShapeCustom import *
import copy

from dataStructure import plane
from globalSetting import *
import cv2
from threading import Thread, current_thread


material_glass = Material(roughness=0.0, transmission=0.95, ior=1.5)
material_ref = Material(roughness=0.005, transmission=0.0, ior=1)#不透光的折射率用不上
material_receiver = Material(roughness=0.0, transmission=0.0, ior=1) # 光线碰上接收器只记录能量，不产生反映，追踪结束
# count_mirror=0
# count_ref=0
# count_receiver=0

up_mirror = plane(h=0,
                  a=vec([6.25, 0., 8]),
                  d=vec([-6.25, 0., 8]),
                  c=vec([-6.25, 0., -8]),
                  b=vec([6.25, 0., -8]),
                  normal=vec([0., 1., 0.]),
                  mtl=material_glass)

bottom_mirror = plane(h=-0.03,
                      a=vec([6.25, -0.03, 8]),
                      d=vec([-6.25, -0.03, 8]),
                      c=vec([-6.25, -0.03, -8]),
                      b=vec([6.25, -0.03, -8]),
                      normal=vec([0., 1., 0.]),
                      mtl=material_ref)

receiver = plane(h=8.0,
                 a=vec([2., 9., 16.]),
                 b=vec([-32., 9., 16.]),
                 c=vec([-32., 7., -16.]),
                 d=vec([2., 7., -16.]),
                 normal=vec([0, -1, 0]),
                 receiver=1,
                 mtl=material_receiver)

LCD_Pad = plane(h=0,
                a=vec([1, 5., 2]),
                b=vec([1, 5., -2]),
                c=vec([5, 3., -2]),
                d=vec([5, 3., 2]),
                normal=vec([-1., -2., 0.]) / np.linalg.norm(vec([-1., -2., 0.])),
                mtl=None)

# new_camera = Camera(pos = vec([]),
#                     front = vec([]),
#                     up = vec([]),
#
#
# )

def check_receiver(obj1:plane, obj2:plane):
    if obj1.a == obj2.a and obj1.b == obj2.b and obj1.c == obj2.c and obj1.d == obj2.d and obj1.normal == obj2.normal:
        return True
    else:
        return False

def intersect_obj(p, ray):
    obj = [up_mirror, bottom_mirror, receiver]

    max_sdf = 1000000.0
    o = plane(sdf=max_sdf)
    for i in range(len(obj)):
        oi = obj[i]
        oi.sdf = oi.planeSDF(p=p)
        # 条件1：np.dot(startpoint, normal) 与 np.dot(dir, normal)符号相反
        # 条件2：sdf在所有obj中最小

        # print(oi.normal)
        # print(p)
        # print(np.dot(p, oi.normal))
        # print(oi_sdf)
        # if np.dot(oi.normal)
        if abs(oi.sdf) < abs(o.sdf) and (oi.checkSide(p)*np.sign(np.dot(oi.normal, ray.Direction))) < 0:
            o = oi
    return o


def rayCast(ray):
    # 光线步进求交
    hitRecord = intersect(pos=ray.Ray_Origin, distance=TMIN, obj=None, flag=False)

    for _ in range(Max_Step):

        hitRecord.position = ray.cal_pos(hitRecord.distance)
        hitRecord.obj = intersect_obj(hitRecord.position, ray)

        dis = abs(hitRecord.obj.sdf)
        if dis < Judge_hit: # 小于判定距离即为相交
            hitRecord.flag = True
            break

        hitRecord.distance += dis # 没相交就继续传播
        if hitRecord.distance > TMAX: # 大于最大传播距离就break
            break
    return hitRecord # 返回碰撞记录



def rayTrace(ray):
    global count_ref, count_mirror, count_receiver
    # receiver =
    for i in range(Max_Trace): # 最多反射Max_Trace次
        hitRecord = rayCast(ray)
        # if hitRecord.obj.mtl == material_ref:
        #     count_ref += 1
        # elif hitRecord.obj.mtl == material_glass:
        #     count_mirror+=1
        # elif hitRecord.obj.mtl == material_receiver:
        #     count_receiver += 1
        # light_quality = 1/50
        # inv_pdf = exp(float(i) * light_quality)
        # roulette_prob = 1.0 - (1.0 / inv_pdf)
        #
        # if np.random.random() < roulette_prob:
        #     break

        if not hitRecord.flag:
            break
        if hitRecord.obj.receiver == 1: # 与接受器相交，接收器对应像素+=能量，追踪结束

            # upx, lowx = normal_discrete(100, hitRecord.hit_pos[0])
            # upy, lowy = normal_discrete(100, hitRecord.hit_pos[0])

            normal_discrete(hitRecord.obj, hitRecord.position, ray)
            # RECEIVER[RECEIVER_WIDTH*dis_xlow:RECEIVER_WIDTH*dis_xup, RECEIVER_HEIGHT*dis_zlow:RECEIVER_HEIGHT*dis_zup] += ray.radiance

            # RECEIVER[lowx:upx, lowz:upz] += ray.Radiance
            break
        # 与镜面相交，继续追踪，镜面对应像素+=能量 继续追踪，pbr计算折射和反射
        normal = hitRecord.obj.normal
        ray = PBR(ray, hitRecord, normal)
    return ray


def hemi_sample(hit_pos, roughness):
    # theta = clamp(abs(np.random.normal(0, roughness, 1)[0])*PI / 2, 0, PI / 2)
    # phi = np.random.uniform(0, 2*PI, 1)[0]
    # x_ori, y_ori, z_ori = hit_pos[0], hit_pos[1], hit_pos[2]
    x_normal = np.random.normal(0, roughness, 1)[0]
    z_normal = np.random.normal(0, roughness, 1)[0]
    temp =clamp(sqrt(x_normal**2 + z_normal**2), 0, 1)
    # if sqrt(temp) >= 1 or x_normal > 1 or z_normal > 1:

    y_normal = sqrt(abs(1 - temp))
    new_normal = vec([x_normal, y_normal, z_normal])
    new_normal = new_normal / np.linalg.norm(new_normal)
    # print(new_normal)

    return new_normal


def PBR(ray, record, normal) -> Ray:
    roughness = record.obj.mtl.roughness  # 获取粗糙度
    # metallic = record.obj.mtl.metallic  # 获取金属度
    transmission = record.obj.mtl.transmission  # 获取透明度
    ior = record.obj.mtl.ior  # 获取折射率

    I = ray.Direction  # 入射方向
    # 将材质的切线空间法线转换到世界空间
    N = copy.deepcopy(normal)  # 法线方向
    N = hemi_sample(N, roughness)
    V = reflect(I, N)  # 反射方向


    NoV = dot(N, V)
    outer = sign(NoV)  # 大于零就是穿入物体，小于零是穿出物体
    # if outer > 0:
    #     transmission = 1
    if np.random.random() < transmission:  # 折射部分 如果是从镜面射出来就全透射
        # if outer > 0:
        # eta = ior / ENV_IOR
        # else:
        eta = ENV_IOR / ior # 折射率之比
        eta = pow(eta, outer)  # 更改折射率之比
        # print(N, outer)
        N *= outer  # 如果是穿出物体表面，就更改法线方向
        NoI = NoV
        k = 1.0 - eta * eta * (1.0 - NoI * NoI)  # 这个值如果小于 0 就说明全反射了

        F0 = (eta - 1) / (eta + 1)  # 基础反射率
        F0 *= F0
        F = fresnel_schlick(NoV, F0)
        # N = hemi_sample(N, roughness)  # 根据粗糙度抖动法线方向

        # k < 0 为全反射
        if np.random.random() < F and outer > 0 or k < 0:
            ray.Direction = reflect(I, N)  # 反射
            ray.Radiance = (1-transmission)*ray.Radiance
            # ray.Direction = ray.Direction
        else:
            # ray.direction = refract(I, N, eta)    # 折射
            ray.Direction = eta * I - (eta * NoI + sqrt(k)) * N
            ray.Radiance = transmission * ray.Radiance
    else:
        # F = fresnel_schlick_roughness(NoI, 0.04, roughness)
        # if np.random.random() < F:  # 反射部分
        ray.Direction = reflect(I, N)  # 平面反射
        # else:  # 漫反射部分
        #     ray.direction = hemi_sample(N)  # 半球采样

    # N = hemispheric_sampling_roughness(N, 0)
    # ray.direction = reflect(I, N)

    ray.Ray_Origin = record.position  # 更新光线起点
    # ray.color.rgb *= record.obj.mtl.albedo  # 更新光的颜色

    return ray



# def pbr(ray, hitRecord, normal):
#     roughness = hitRecord.obj.roughness
#     # normal = hitRecord.obj.normal
#     # out_dir = dot(ray.direction, normal) < 0.0
#     # normal *= 1.0 if out_dir else -1.0
#     refract = hitRecord.obj.mtl.ior
#     transmission = hitRecord.obj.mtl.transmission
#
#     I = ray.Direction
#     R = -ray.Direction # 反射方向
#     N = normal
#
#     Nov = dot(N, R)
#
#     if np.random.random() < transmission: # 折射
#         eta = ENV_IOR / refract
#         out_dir = sign(Nov)# 大于0穿入，小于0穿出
#         eta = pow(eta, out_dir)
#
#         N *= out_dir # 穿出就更改法线方向
#
#         NoI = -Nov
#         k = 1.0 - eta * eta * (1.0 - NoI * NoI)  # 这个值如果小于 0 就说明全反射了
#
#         F0 = (eta - 1) / (eta + 1)  # 基础反射率
#         F0 *= F0
#         F = fresnel_schlick(Nov, F0)
#         # N = hemi_sample(hitRecord.hit_pos, roughness)
#         # N = N
#         if np.random.random() < F and out_dir > 0 or k < 0:
#             # 反射
#             ray.Direction = reflect(I, N)
#
#         else:
#             # 折射
#             ray.Direction = eta * I - (eta * NoI + sqrt(k)) * N
#     else:
#         # F = 1
#         # if np.random.random() < F:  # 反射部分
#         N = hemi_sample(hitRecord.hit_pos, roughness)
#         ray.Direction = reflect(I, N)  # 平面反射
#         # else:  # 漫反射部分
#         #     ray.direction = hemispheric_sampling(N)  # 半球采样
#
#     ray.Ray_Origin = hitRecord.position
#     # ray.Radiance =
#     return ray

def main():
    # 生成光线
    # for i in range(20000):
    #     # ray = get_ray(lightSourceType='parallel', middle=vec([6.25, 6.25, 0]), radius=2)
    #     ray = get_ray(lightSourceType='zebra', middle=vec([6.25, 6.25, 0]), radius=1, step=0.2)
    #     next_ray = rayTrace(ray)

    # light_pattern[1, 1] = 1
    # light_pattern[2, 1] = 1
    #
    # emission = np.where(np.matrix(light_pattern) == 1)  # 发光像素坐标
    # for i in range(len(emission)):
    LP = getLightingPattern(np.zeros((10, 10)), stride=2)
    ray_list = get_ray('lightPad', middle=vec([6.25, 6.25, 0]), LCD_Pad=LCD_Pad,radius=None, LightPattern=LP)
    for item_ray in ray_list:
        next_ray = rayTrace(item_ray)
    visualization(RECEIVER, LP)

main()
print(count_mirror, count_ref, count_ref)
print(RECEIVER.sum())
print(RECEIVER)






























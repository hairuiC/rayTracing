

import numpy as np
from numpy.core._multiarray_umath import dot

from globalSetting import *



class Ray:
    def __init__(self, origin, direction, radiance=1):
        self.Ray_Origin = origin # vec3
        self.Direction = direction # vec3
        self.Radiance = radiance # float 能量
    def cal_pos(self, t):
        return self.Ray_Origin + t * self.Direction

    def cal_radiance(self, t, att):
        curr_pos = self.cal_pos(t)
        distance = np.linalg.norm(curr_pos - self.Ray_Origin)
        att =  1.0 / (att_kc + att_kl * distance + att_kq * (distance * distance))
        self.Radiance *= att
class Material:
    def __init__(self, roughness, transmission, ior):
        # self.albeto = albeto
        self.roughness = roughness
        self.transmission = transmission
        self.ior = ior
        # self.normal = normal
class Camera:
    def __init__(self, pos, lookat, up, fov, aspect):
        self.pos = pos
        self.lookat = lookat
        self.up = up
        self.fov = fov
        self.aspect = aspect
    def cal_imageplane(self):
        unit_dir = self.lookat / np.linalg.norm(self.lookat)
        plane_mid = self.pos + 1 * unit_dir

        theta = self.fov * PI / 180
        half_height = np.tan(theta / 2)
        half_width = self.aspect * half_height

        w = self.pos - self.lookat
        u = np.cross(self.up, w)
        v = np.cross(w, u)

        a = plane_mid + half_height * v
        b = a + 2 * half_width * u
        c = plane_mid - half_height * v
        d = c + 2 * half_width * u

        image_plane = plane(a=vec([6.25, 0., 8]),
                          b=vec([-6.25, 0., 8]),
                          c=vec([-6.25, 0., -8]),
                          d=vec([6.25, 0., -8]),
                          normal=vec([0., 1., 0.]),
                            mtl=None)
     # horizontal = 2 * half_width * u
        # vertical = 2 * half_height * v

class intersect:
    def __init__(self, pos, distance, obj, flag=False):
        self.position = pos
        self.distance = distance
        self.obj = obj
        self.flag = flag

class plane:
    def __init__(self, h=0, a=ini_vec, b=ini_vec, c=ini_vec, d=ini_vec, normal=ini_vec,sdf=0,mtl:Material=None, receiver=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.sdf = sdf
        self.receiver = receiver
        self.normal = normal
        self.mtl = mtl
        self.h = h

        x1, y1, z1 = a[0], a[1], a[2]
        x2, y2, z2 = b[0], b[1], b[2]
        x3, y3, z3 = c[0], c[1], c[2]

        self.A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
        self.B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
        self.C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        self.D = -x1 * (y2 * z3 - y3 * z2) - x2 * (y3 * z1 - y1 * z3) - x3 * (y1 * z2 - y2 * z1)
        # self.transmission = 1.3

    def planeSDF(self, p, h):
        return np.dot(p, self.normal) + h
    def checkSide(self, p):
        # A = self.normal[0]
        # B = self.normal[1]
        # C = self.normal[2]
        D = -np.dot(self.normal, self.d)
        return np.sign(np.dot(self.normal, p) + D)
        # 小于0与法向量不在同一侧，大于0与法向量在同一侧



    def SDF(self, p):

        ba = self.b - self.a
        cb = self.c - self.b
        dc = self.d - self.c
        ad = self.a - self.d

        pa = p - self.a
        pb = p - self.b
        pc = p - self.c
        pd = p - self.d

        normal = np.cross(ba, ad)
        temp = sqrt(sign(dot(np.cross(ba, normal), pa)) +
                    sign(dot(np.cross(cb, normal), pb)) +
                    sign(dot(np.cross(dc, normal), pc)) +
                    sign(dot(np.cross(ad, normal), pd)))
        if temp < 3.0:
            return min(min(min(
                dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
                dot2(dc * clamp(dot(dc, pc) / dot2(dc), 0.0, 1.0) - pc)),
                dot2(ad * clamp(dot(ad, pd) / dot2(ad), 0.0, 1.0) - pd))
        else:
            return dot(normal, pa) * dot(normal, pa) / dot2(normal)

    # def calPos(self):






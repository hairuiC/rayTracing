# 定制光源：1、形状 亮度 分布 光源位置
#         2、光线数目 点光源/平行光
from utils import *
class Light:
    def __init__(self, location, num_light, radiance):
        self.location = location
        self.num_light = num_light
        self.radiance = radiance



class punctualLight(Light):
    def __init__(self):
        self.pos = self.location


class LCDPad(Light):
    def __init__(self, Width, Height, loc_on):
        pass



class parallelLight(Light):
    def __init__(self, lightRadius):
        # self.orientation = orientation
        self.lightRadius = lightRadius
        self.origin, self.direction = get_ray('parallel', self.location, self.lightRadius)

# class zebraLight(Light):
#     # 条纹光：一条发光一条不发光,发光间隔由step确定
#     def __init__(self, lightWidth, step:float):
















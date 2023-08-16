import numpy as np
from utils import *

# print(np.random.rand(100, 2))
# a = np.matrix([[1, 0, 0],
#      [0, 1, 0],
#      [1, 0, 1]])
#
# print(np.argwhere(a == 1))

# z = 0

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    print(a, b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
# randx, y = np.random.rand(2)
# print(randx, y)
mat = np.zeros((10, 10))
getLightingPattern(mat = mat, stride=2, padding=0)


# a = vec([0, -1, 0])
# b = vec([-1, 0, 0])
#
# rota = rotation_matrix_from_vectors(a, b)
# a = a.reshape(3, 1)
# print(a)
# print(rota * a)
# print(a)
# cos_thet
# a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# sin_theta = np.cross(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# Rx = np.array([[1, 0, 0],[0, cos_theta, -sin_theta],[0, sin_theta, cos_theta]])

# print(Rx)
# R = get_rotation(a, b)


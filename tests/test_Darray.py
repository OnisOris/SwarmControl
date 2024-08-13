from ThreeDTool import loxodrome
from SwarmControl.swarmsys import Darray
import numpy as np
from numpy import cos, sin
import random


class TestDarray:
    def test_goto(self):
        darr = Darray()
        darr.create_square_array(center_point=np.array([0, 0, 0]))
        traj = loxodrome(R=10, step=1, point_n=np.array([0, 0, 10]))
        for point in traj:
            darr.goto(point)
        assert np.allclose(darr.body.point, traj[-1], 1e-8)

    def test_goto_every_drone(self):
        darr = Darray()
        darr.create_square_array(center_point=np.array([0, 0, 0]))
        traj = loxodrome(R=10, step=1, point_n=np.array([0, 0, 10]))
        for point in traj:
            darr.goto(point)
        assert np.allclose(darr[0].body.point, darr[0].begin_point + darr.body.point, 1e-8)

    def test_create_square_array(self):
        test_matrix = np.array([[0., -10., 1.],
                                [20., -10., 1.],
                                [0., 10., 1.],
                                [20., 10., 1.]
                                ])
        arr = Darray()
        arr.create_square_array(number_of_drones=2, center_point=np.array([10, 0, 0]))
        eq_matrix = np.array([[0, 0, 0]])
        for drone in arr:
            eq_matrix = np.vstack([eq_matrix, drone.body.point])
        eq_matrix = eq_matrix[1:]
        assert np.allclose(test_matrix, eq_matrix, 1e-8)

    def test_Darray_rot_x(self):
        rot_point = np.array([40, 40, 0])
        point = np.array([0, 0, 0])
        darr = Darray(xyz=point, orientation=np.array([[1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 1]]))
        darr.create_square_array(center_point=np.array([0, 0, 0]))
        rand_drone = darr[random.randint(0, np.shape(darr.drones)[0] - 1)]
        # Найдем вектор смещения дрона относительно центра массива по векторному вычитанию
        vector_s = rand_drone.body.point - darr.body.point
        angle = np.pi / 4
        darr.rot_x(angle, rot_point=rot_point, apply=True)
        rotate_x = np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        # Здесь проверяется положение центра массива и положение каждого дрона в данном массиве, vector_s также необходимо
        # умножить на матрицу поворота
        assert np.allclose(darr.body.point, (point - rot_point).dot(rotate_x) + rot_point,
                           1e-8) and np.allclose(rand_drone.body.point,
                                                 (point - rot_point).dot(rotate_x) + rot_point + vector_s.dot(rotate_x),
                                                 1e-8)

    def test_Darray_rot_y(self):
        rot_point = np.array([40, 40, 0])
        point = np.array([0, 0, 0])
        darr = Darray(xyz=point, orientation=np.array([[1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 1]]))
        darr.create_square_array(center_point=np.array([0, 0, 0]))
        rand_drone = darr[random.randint(0, np.shape(darr.drones)[0] - 1)]
        # Найдем вектор смещения дрона относительно центра массива по векторному вычитанию
        vector_s = rand_drone.body.point - darr.body.point
        angle = np.pi / 4
        darr.rot_y(angle, rot_point=rot_point, apply=True)
        rotate_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        # Здесь проверяется положение центра массива и положение каждого дрона в данном массиве, vector_s также необходимо
        # умножить на матрицу поворота
        assert np.allclose(darr.body.point, (point - rot_point).dot(rotate_y) + rot_point,
                           1e-8) and np.allclose(rand_drone.body.point,
                                                 (point - rot_point).dot(rotate_y) + rot_point + vector_s.dot(rotate_y),
                                                 1e-8)

    def test_Darray_rot_z(self):
        rot_point = np.array([40, 0, 0])
        point = np.array([0, 0, 0])
        darr = Darray(xyz=point, orientation=np.array([[1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 1]]))
        darr.create_square_array(center_point=np.array([0, 0, 0]))
        rand_drone = darr[random.randint(0, np.shape(darr.drones)[0] - 1)]
        # Найдем вектор смещения дрона относительно центра массива по векторному вычитанию
        vector_s = rand_drone.body.point - darr.body.point
        angle = np.pi / 4
        darr.rot_z(angle, rot_point=rot_point, apply=True)
        rotate_z = np.array([[cos(angle), -sin(angle), 0],
                             [sin(angle), cos(angle), 0],
                             [0, 0, 1]])
        # Здесь проверяется положение центра массива и положение каждого дрона в данном массиве, vector_s также необходимо
        # умножить на матрицу поворота
        assert np.allclose(darr.body.point, (point - rot_point).dot(rotate_z) + rot_point,
                           1e-8) and np.allclose(rand_drone.body.point,
                                                 (point - rot_point).dot(rotate_z) + rot_point + vector_s.dot(rotate_z),
                                                 1e-8)

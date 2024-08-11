from __future__ import annotations
import time
from typing import Any
import threading
import loguru
import numpy as np
import ThreeDTool as tdt
from ThreeDTool import Line_segment, Polygon, Line, Curve
from numpy import cos, sin, ndarray, dtype, pi
from pioneer_sdk import Pioneer
from body import Body
from dspl import Dspl
from config import CONFIG
import pygame
from icecream import ic
import pandas as pd

def set_barycenter(array: ndarray | list) -> ndarray:
    """
    Вычисляет барицентр фигуры
    :return: None
    """
    if isinstance(array, list):
        array = np.array(array)
    return array.mean(axis=0)


def rot_x(vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    """
    Функция вращает входные вектора вокруг оси x, заданной векторами-столбцами. Положительным вращением считается
    по часовой стрелке при направлении оси к нам.
    :param vector: Входной вращаемый вектор
    :param angle: угол вращения, заданный в радианах
    :return: np.ndarray
    """
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    rot_vector = vector.dot(rotate_x)
    return rot_vector


def rot_y(vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    """
    Функция вращает входные вектора вокруг оси y, заданной векторами-столбцами. Положительным вращением считается
    по часовой стрелке при направлении оси к нам.
    :param vector: Входной вращаемый вектор
    :param angle: угол вращения, заданный в радианах
    :return: np.ndarray
    """
    rotate_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    rot_vector = vector.dot(rotate_y)
    return rot_vector


def opposite_vectors(v1, v2) -> bool:
    """
    Функция проверяет, противоположны ли векторы
    :param v1: Вектор 1
    :type v1: np.ndarray
    :param v2: Вектор 2
    :type v2: np.ndarray
    :return: bool
    """
    v1 = np.array(v1) / np.linalg.norm(v1)
    v2 = np.array(v2) / np.linalg.norm(v2)
    if np.sum(v1 + v2) == 0.0:
        return True
    else:
        return False


def rot_z(vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    """
    Функция вращает входные вектора вокруг оси z, заданной векторами-столбцами. Положительным вращением считается
    по часовой стрелке при направлении оси к нам.
    :param vector: Вращаемые вектор-строки, упакованные в матрицу nx3
    :param angle: угол вращения, заданный в радианах
    :return: np.ndarray
    """
    vector = np.array(vector)
    rotate_z = np.array([[cos(angle), -sin(angle), 0],
                         [sin(angle), cos(angle), 0],
                         [0, 0, 1]])
    if vector.shape == (3,) or vector.shape == (1, 3):

        rot_vector = vector.dot(rotate_z)
        return rot_vector
    elif vector.shape == (2,) or vector.shape == (1, 2):
        vector = np.hstack([vector, 0])
        rot_vector = vector.dot(rotate_z)
        return rot_vector[0:2]


def angle_from_vectors(vector1: ndarray, vector2: ndarray) -> ndarray:
    """
    Функция возвращает угол между двумя векторами в радианах, заданными в n_мерном пространстве
    :param vector1: Первый n-мерный вектор-строка
    :type vector1: ndarray
    :param vector2: Второй n-мерный вектор-строка
    :type vector2: ndarray
    :return: ndarray
    """
    from math import sqrt
    from ThreeDTool import null_vector
    if null_vector(vector1) or null_vector(vector2):
        raise Exception("Вектора не не могут быть равны нулю")
    cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arctan2(-sqrt(1 - cos ** 2), cos)
    return angle


def rot_v(vector: list | np.ndarray, angle: float | int, axis: list | np.ndarray) -> np.ndarray:
    """
    Функция вращает входные вектора вокруг произвольной оси, заданной векторами-столбцами. Положительным вращением
    считается по часовой стрелке при направлении оси к нам.
    :param axis: Вектор произвольной оси, вокруг которой происходит вращение
    :type axis: list | np.ndarray
    :param vector: Вращаемые вектор-строки, упакованные в матрицу nx3
    :type vector: list | np.ndarray
    :param angle: Угол вращения в радианах
    :type angle: float | int
    :return: np.ndarray
    """
    axis = normalization(axis, 1)
    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    t = 1 - c
    rotate = np.array([
        [t * x ** 2 + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y ** 2 + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z ** 2 + c]
    ])
    rot_vector = np.dot(vector, rotate)
    return rot_vector


def normalization(vector: list | ndarray, length: int | float = 1) -> ndarray:
    """
    Функция возвращает нормированный вектор заданной длины
    :param vector: Вектор
    :type vector: list | ndarray
    :param length: Длина вектора
    :type length: int
    """
    return np.array(vector) / np.linalg.norm(vector) * length


def euler_rotate(vector: list | np.ndarray, alpha: float, beta: float, gamma: float) -> ndarray:
    """
    Функция воспроизводит вращение Эйлера в последовательности z, x, z. Положительным вращением считается
    по часовой стрелке при направлении оси к нам.
    :param vector: Вращаемые вектора размерностью
    :param alpha: Первый угол вращения по оси z, в радианах
    :type alpha: float
    :param beta: Второй угол вращения по оси x, в радианах
    :type alpha: float
    :param gamma: Третий угол вращения по оси z, в радианах
    :type alpha: float
    :return: np.ndarray
    """
    rot_vector = rot_z(rot_x(rot_z(vector, alpha), beta), gamma)
    return rot_vector


class Drone:
    """
    Класс единицы дрона в трехмерном пространстве.
    """

    def __init__(self, point: list | np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]),
                 drone: Pioneer = None,
                 apply: bool = True,
                 joystick_on: bool = False):
        """
        В данной реализации дрон имеет координату point и вектор ориентации в глобальной системе координат
        :param point:
        :param orientation: Ориентация, представляющая собой матрицу 3x3 из единичных векторов-строк. Первый вектор
        задает x, второй y, третий z.
        """
        self.joystick_on = joystick_on

        self.wait_point = False
        self.traj = np.array([0, 0, 0, 0, 0, 0, 0])
        self.xyz_flag = False
        self.body = Body(point, orientation)
        self.trajectory = np.array([])
        self.drone = drone
        self.apply = apply
        self.begin_point = point
        # характеристики дрона
        self.height = CONFIG['height']
        self.length = CONFIG['lenght']
        self.width = CONFIG['width']
        self.rad = np.linalg.norm([self.length / 2, self.width / 2])
        self.active = False  # Флаг, показывающий, что дрон в активном состоянии, т.е меняет свое положение
        self.t = []
        self.occupancy = False
        self.speed_flag = True
        self.t0 = time.time()
        if self.joystick_on:
            pygame.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Название джойстика: {self.joystick.get_name()}")
            print(f"Количество осей: {self.joystick.get_numaxes()}")
            print(f"Количество кнопок: {self.joystick.get_numbuttons()}")

    def stop(self) -> None:
        """
        Функция снимает все флаги для остановки потоков
        :return: None
        """
        self.speed_flag = False
        self.xyz_flag = False
        if self.joystick_on:
            pygame.quit()

    def save_data(self):
        self.stop()
        time.sleep(2)
        self.drone.land()
        time.sleep(8)
        self.drone.disarm()
        ic(self.traj)
        df = pd.DataFrame(self.traj, columns=['x', 'y', 'z', 'Vx', 'Vy', 'Vz', 't'])
        df.to_csv('../../plot_out/data.csv')
        # with open(f'./plot_out/xyz_Vx_Vy_Vz_t.npy', 'wb') as f:
        #     np.save(f, self.traj)

    def get_position(self, filter=False) -> None | list:
        """
        Функция возвращает координату дрона с фильтром компонентов
        :return: list
        """
        from icecream import ic
        if 'LOCAL_POSITION_NED' in self.drone.msg_archive:
            msg_dict = self.drone.msg_archive['LOCAL_POSITION_NED']
            msg = msg_dict['msg']
            if not msg_dict['is_read'].is_set() or (msg_dict['is_read'].is_set()):
                msg_dict['is_read'].set()
                zero_point = np.array([0., 0., 0.])
                if msg._header.srcComponent == 26:
                    point = [msg.x/1000, msg.y/1000, msg.z/1000]
                    # ic(point)
                elif msg._header.srcComponent == 1:
                    point = [msg.x, msg.y, msg.z]
                    # ic(point)
                else:
                    return None
                if filter:
                    if not np.all(np.equal(point, zero_point)):
                        return point
                    else:
                        return None
                else:
                    return point
            else:
                return None
        else:
            return None

    def attach_body(self, body: Body) -> None:
        """
        Функция устанавливает тело для дрона
        :param body: Объект тела, задающий ориентацию и положение в пространстве
        :type body: Body
        :return: None
        """
        self.body = body

    def get_border(self) -> np.ndarray:
        """
        Функция возвращает границу дрона в виде матрицы 4x3, которая является вершинами прямоугольника, вида
        [[x1, y1, z1], ......., [x4, y4, z4]]
        :return: np.ndarray
        """
        return np.array([[self.body.point[0] - self.rad / 2, self.body.point[1] + self.rad / 2, self.body.point[2]],
                         [self.body.point[0] + self.rad / 2, self.body.point[1] + self.rad / 2, self.body.point[2]],
                         [self.body.point[0] + self.rad / 2, self.body.point[1] - self.rad / 2, self.body.point[2]],
                         [self.body.point[0] - self.rad / 2, self.body.point[1] - self.rad / 2, self.body.point[2]]])

    def get_polygon(self) -> Polygon:
        """
        Функция возвращает границу дрона в виде объекта класса Polygon. Граница является вершинами прямоугольника, вида
        [[x1, y1, z1], ......., [x4, y4, z4]]
        :return: Polygon
        """
        return Polygon(self.get_border())

    def goto(self, point: list | np.ndarray | None = None,
             orientation: list | np.ndarray | None = None,
             apply: bool = False,
             wait_point: bool | None = True) -> None:
        """
        Функция перемещает дрон в заданное положение. Если задана точка назначения и вектор ориентации, тогда
        изменится все. Задана только ориентация или только точка, то изменится только нужный параметр.
        Если не задано ничего, то ничего не поменяется.
        :param wait_point: Ждем, пока дрон не прилетит до точки
        :param apply: :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :param orientation: Ориентация, представляющая собой матрицу 3x3 из единичных векторов-строк. Первый вектор
         задает x, второй y, третий z.
        :type orientation: list or np.ndarray or None
        :param point: Точка назначения для дрона
        :type point: list[float, int] or None
        :return: None
        """
        if point is None and orientation is None:
            return
        # Здесь будет код для перемещения дрона, например через piosdk
        old_orinetation = self.body.orientation
        if orientation is not None:
            # Здесь будет функция смены отображения ориентации, как self.trajectory_write()
            self.body.orientation = orientation
        if point is not None:
            self.trajectory_write(self.body.point, point)
            self.body.point = point
        if self.apply & apply:
            self.apply_position(point)
        if self.wait_point:
            self.wait_for_point(self.body.point)

    def send_v(self, v: list | ndarray, ampl: float | int = 1) -> None:
        """
        Функция задает вектор скорости дрону
        :param v: Вектор скорости
        :type v: list | np.ndarray
        :return: None
        """
        v = ampl * rot_z(v, CONFIG['rot_send_U'])
        if v.shape == (2,):
            self.drone.set_manual_speed(v[0], v[1], 1.5, 0)
        elif v.shape == (3,):
            self.drone.set_manual_speed(v[0], v[1], v[2], 0)
        elif v.shape == (4,):
            self.drone.set_manual_speed(v[0], v[1], v[2], v[3])

    def v_while(self) -> None:
        """
        Функция задает цикл while на отправку вектора скорости в body с периодом period_send_v
        :return: None
        """
        if self.joystick_on:
            f = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
            flag_arm = False
            while self.speed_flag:
                pygame.event.pump()
                axes = [round(self.joystick.get_axis(i), 2) for i in range(self.joystick.get_numaxes())]
                if not np.allclose(axes, f, 1e-3):
                    f = axes
                    # ic(axes[4])
                    if f[5] == 1.0:
                        self.send_v([f[0], -f[1], f[2]], 3)
                    elif f[5] == -1.0:
                        self.send_v(self.body.v)
                    if f[4] == 1.0:
                        if not flag_arm:
                            loguru.logger.debug("arm_joystick")
                            self.drone.arm()
                            loguru.logger.debug("takeoff_joystick")
                            self.drone.takeoff()
                        flag_arm = True
                    elif axes[4] == -1.0:
                        if flag_arm:
                            loguru.logger.debug("land_joystick")
                            self.drone.land()
                            loguru.logger.debug("disarm_joystick")
                            self.drone.disarm()
                        flag_arm = False
                    if f[7] == 1:
                        self.save_data()
                time.sleep(CONFIG['period_send_v'])
        else:
            while self.speed_flag:
                self.send_v(self.body.v)
                time.sleep(CONFIG['period_send_v'])

    def set_v(self) -> None:
        """
        Создает поток, который вызывает функцию v_while() для параллельной отправки вектора скорости
        :return: None
        """
        self.t.append(threading.Thread(target=self.v_while))
        self.t[-1].start()

    def speed_change(self, v: list | np.array) -> None:
        if np.shape(v) == (3,):
            self.body.v = v
        elif np.shape(v) == (2,):
            self.body.v = np.hstack([v, 0])

    def trajectory_write(self, previous_point: list | np.ndarray, current_point: list | np.ndarray) -> None:
        """
        Функция сохраняет траекторию движения по точкам во внутренний массив с объектами класса Line_segment
        :param previous_point: Начальная точка траектории
        :type previous_point: list | np.ndarray
        :param current_point: Текущая точка траектории
        :type current_point:  list | np.ndarray
        :return: None
        """
        segment = Line_segment(point1=previous_point, point2=current_point)
        segment.color = 'orange'
        self.trajectory = np.hstack((self.trajectory, segment))

    def xyz_while(self):
        while self.xyz_flag:
            coord = self.get_position(filter=False)
            if coord is not None:
                self.body.real_point = np.array(coord)
                if CONFIG['trajectory_write']:
                    t = time.time() - self.t0
                    stack = np.hstack([self.body.real_point, self.body.v[0:3], t])
                    self.traj = np.vstack([self.traj, stack])
            time.sleep(0.05)

    def set_coord_check(self) -> None:
        """
        Функция отправляет команду на приземление дрона
        :return: None
        """
        # if self.apply:
        self.t.append(threading.Thread(target=self.xyz_while))
        self.t[-1].start()

    def apply_position(self,
                       point: list | np.array | None = None,
                       rad: bool = False,
                       angle=0) -> None:
        """
        Данная функция отправляет дронам изменившеюся ориентацию и позицию в orientation и в point
        :return: None
        """
        import math
        if self.drone is not None:
            # if rad:
            #     yaw = math.atan2(self.body.orientation[0][1],
            #                      self.body.orientation[0][0])
            # else:
            #     # yaw = math.atan2(self.body.orientation[0][1],
            #     #                   self.body.orientation[0][0]) * 180 / np.pi
            #     yaw = angle * 180 / np.pi
            # yaw = angle * 180 / np.pi
            if point is None:
                self.drone.go_to_local_point(self.body.point[0],
                                             self.body.point[1],
                                             self.body.point[2],
                                             # перед yaw стоит минус, так как дроны вращаются не в ту сторону в pio_sdk
                                             yaw=0)
            else:
                point = rot_z(point, CONFIG['rot_send_U'])
                self.drone.go_to_local_point(point[0],
                                             point[1],
                                             point[2],
                                             # перед yaw стоит минус, так как дроны вращаются не в ту сторону в pio_sdk
                                             yaw=0)

    def euler_rotate(self, alpha: float, beta: float, gamma: float, apply: bool = False) -> None:
        """
        Функция воспроизводит вращение Эйлера в последовательности z, x, z. Положительным вращением считается
        по часовой стрелке при направлении оси к нам.
        :param alpha: Первый угол вращения по оси z, в радианах
        :type alpha: float
        :param beta: Второй угол вращения по оси x, в радианах
        :type alpha: float
        :param gamma: Третий угол вращения по оси z, в радианах
        :type alpha: float
        :param apply: Параметр, указывающий, отправлять ли данные в дрон
        :return: None
        """
        self.rot_v(alpha)
        self.rot_x(beta)
        self.rot_v(gamma)
        if self.apply & apply:
            self.apply_position()

    def rot_x(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси y. Положительным вращением считается
        по часовой стрелке при направлении оси к нам.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.trans(rot_x, angle=angle, rot_point=rot_point, apply=apply)

    def rot_y(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси y. Положительным вращением считается
        по часовой стрелке при направлении оси к нам.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.trans(rot_y, angle=angle, rot_point=rot_point, apply=apply)

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = False) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси z. Положительным вращением считается
        по часовой стрелке при направлении оси к нам.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.trans(rot_z, angle=angle, rot_point=rot_point, apply=apply)

    def rot_v(self, angle: float | int,
              rot_point: np.ndarray = np.array([0, 0, 0]),
              axis: list | np.ndarray = np.array([0, 0, 1]),
              apply: bool = False) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси z. Положительным вращением считается
        по часовой стрелке при направлении оси к нам.
        :param axis:
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.body.orientation = rot_v(self.body.orientation, angle, axis)
        new_point = rot_v(self.body.point - rot_point, angle, axis) + rot_point
        self.trajectory_write(self.body.point, new_point)
        self.body.point = new_point
        if self.apply & apply:
            self.apply_position()

    def trans(self, func, angle: float | int,
              rot_point: np.ndarray = np.array([1, 0, 0]),
              apply: bool = False) -> None:
        """
        Данная функция преобразует координаты и ориентацию дрона с помощью функции преобразования func
        :param func: Функция преобразования ориентации
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        old_orientation = self.body.orientation
        self.body.orientation = func(self.body.orientation, angle)
        new_point = func(self.body.point - rot_point, angle) + rot_point
        self.trajectory_write(self.body.point, new_point)
        self.body.point = new_point
        if self.apply & apply:
            self.apply_position(old_orientation, angle=angle)

    def self_show(self) -> None:
        """
        Функция отображает дрон на трехмерном графике
        :return: None
        """
        dp = Dspl([self.body], create_subplot=True, qt=True)
        self.body.show(dp.ax)
        dp.show()

    def show(self, ax) -> None:
        """
        Функция отображает дрон на трехмерном графике, принимая внешний ax от matplotlib
        :return: None
        """
        self.body.show(ax)

    def show_trajectory(self, ax) -> None:
        """
        Функция отображает траектории дрона на трехмерном графике, принимая внешний ax от matplotlib
        :return: None
        """
        for traj in self.trajectory:
            traj.show(ax)

    def arm(self) -> None:
        """
        Функция отправляет команду на включение двигателей на дрон
        :return: None
        """
        if self.apply:
            self.t.append(threading.Thread(target=self.drone.arm, args=()))
            self.t[-1].start()

    def disarm(self) -> None:
        """
        Функция отправляет команду на отключение двигателей на дрон
        :return: None
        """
        if self.apply:
            self.t.append(threading.Thread(target=self.drone.disarm, args=()))
            self.t[-1].start()

    def takeoff(self, height: float | int = 1.8) -> None:
        """
        Функция отправляет команду на взлет дрона
        :return: None
        """
        def smart_takeoff():
            self.drone.takeoff()
            position = None
            while position is None:
                position = self.get_position()
            x, y, z = position
            self.goto([x, y, height])

        if self.apply:
            self.t.append(threading.Thread(target=smart_takeoff(), args=()))
            self.t[-1].start()

    def land(self) -> None:
        """
        Функция отправляет команду на приземление дрона
        :return: None
        """
        if self.apply:
            self.t.append(threading.Thread(target=self.drone.land, args=()))
            self.t[-1].start()

    def check_collision(self, drone: Drone, map_object) -> bool:
        line_s = Line_segment(point1=self.body.point, point2=drone.body.point)
        for border in map_object.borders:
            for segment in border.get_line_segments():
                p = tdt.point_from_segment_segment_intersection(line_s, segment)
                if p is not None:
                    return True
        return False

    def calculate_path(self, target_point, map_object: Map):
        """

        :param target_point:
        :param map_object:
        :return:
        """
        target_point[2] = map_object.z
        line_s = Line_segment(point1=self.body.point, point2=target_point)  # + R/2
        full_vec = line_s.coeffs().reshape(2, 3)
        full_vec2 = tdt.rotation_full_vector_relative_point_axis(full_vec, np.pi / 2, full_vec[0], axis=[0, 0, 1])
        first_point = self.body.point + tdt.normalization(full_vec2[1], self.rad)
        second_point = self.body.point - tdt.normalization(full_vec2[1], self.rad)
        rectangle = tdt.rectangle_from_three_points(first_point, second_point, target_point)
        polygons = np.array([])
        points = np.array([[0, 0, 0]])
        # Необходимо взять объекты без границы самого дрона
        map_objects = map_object.get_border_without_border_drone(drone=self)

        dp_debug = Dspl(map_objects)
        dp_debug.show()
        loguru.logger.debug(f"map ---> {map_objects}")
        for border in map_objects:
            p = rectangle.polygon_analyze(border)
            if p is not None:
                # loguru.logger.debug((p))
                polygons = np.hstack([polygons, border])
                points = np.vstack([points, p])
        if np.shape(points) != (3,):
            points = points[1:]
        points = np.vstack([points, target_point])
        # loguru.logger.debug(points)
        points_int = tdt.Points(np.array(points), color='red', s=10)
        dp = tdt.Dspl(np.hstack([points_int, polygons, line_s, rectangle]))
        dp.show()
        # loguru.logger.debug(points_int)

    def info(self):
        return f"x: {self.body.point[0]}, y: {self.body.point[1]}, z: {self.body.point[2]}"

    def wait_for_point(self, point: list | ndarray, accuracy: float = 1e-2):
        import time
        self.occupancy = True
        loguru.logger.debug(f"point = {point}")
        while self.occupancy:
            k = self.get_position()
            loguru.logger.debug(f"point = {point} k = {k}")
            if isinstance(k, list):
                self.occupancy = not np.allclose(np.array(k), self.body.point, accuracy)
            time.sleep(0.5)


class Darray:
    """
    Данный класс хранит в себе массив с дронами и имеет методы для их общего управления
    """

    def __init__(self, drones: list[Drone] | np.ndarray[Drone] = None,
                 xyz: list | np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]),
                 apply: bool = False,
                 axis_length: int | None = 5,
                 xy_inverse: bool = False):
        """
        :param drones: Массив с дронами
        :type drones: list[Drone] or np.ndarray[Drone]
        :param xyz: Координата массива дронов (может быть любым, но обычно это центр)
        :type xyz: list | np.ndarray
        :param orientation: Ориентация, представляющая собой матрицу 3x3 из единичных векторов-строк. Первый вектор
         задает x, второй y, третий z.
        :type orientation: np.ndarray
        :param apply: Отправлять ли данные в дроны
        :type apply: bool
        :param axis_length: длина осей для отображения на сцене
        :type axis_length: int | None

        """
        if drones is None:
            self.drones = []
        else:
            self.drones = drones
        self.apply = apply
        self.trajectory = np.array([])
        self.body = Body(xyz, orientation)
        self.body.length = axis_length
        self.occupancy = False
        self.xy_inverse = xy_inverse

    def __getitem__(self, item):
        return self.drones[item]

    def __iter__(self):
        return iter(self.drones)

    def create_square_array(self, defining_points: list | np.ndarray = np.array([[-10, -10], [10, 10]]),
                            number_of_drones_in_row: int = 4,
                            center_point: np.ndarray = np.array([0, 0, 0]),
                            pio_drones=None) -> None:
        """
        Функция генерирует квадратный массив из дронов
        :param defining_points: Две точки, задающие прямоугольник
        :type defining_points: list or np.ndarray
        :param number_of_drones_in_row: Количество дронов
        :type number_of_drones_in_row: int
        :param center_point: Центральная точка формирования массива дронов
        :type center_point: np.ndarray
        :param pio_drones: Входящие объекты дронов
        :type pio_drones: list | np.ndarray
        :return: None
        """
        self.body.point = center_point
        x = np.linspace(defining_points[0][0], defining_points[1][0], number_of_drones_in_row) + center_point[0]
        y = np.linspace(defining_points[0][1], defining_points[1][1], number_of_drones_in_row) + center_point[1]
        z = np.ones(number_of_drones_in_row ** 2)
        x_m, y_m = np.meshgrid(x, y)
        x_m = x_m.reshape(-1)
        y_m = y_m.reshape(-1)
        points = np.vstack((x_m, y_m, z)).T
        drones = np.array([])
        for point in points:
            drones = np.hstack((drones, Drone(point, self.body.orientation)))
        self.drones = drones
        if pio_drones is not None:
            for i, drone in enumerate(self.drones):
                drone.drone = pio_drones[i]

    def create_local_array(self, pio_drones=None) -> None:
        """
        Функция генерирует массив из дронов по их начальным координатам
        :param pio_drones: Входящие объекты дронов
        :type pio_drones: list | np.ndarray
        :return: None
        """
        if pio_drones is None:
            raise Exception("Дроны должны быть поданы в данную функцию")
        points = np.array([[0, 0, 0]])
        drones = np.array([])
        for pioner in pio_drones:
            while True:
                point = pioner.get_local_position_lps()
                if point is not None:
                    if self.xy_inverse:
                        y, x, _ = point
                        point = [x, y, 1]
                    else:
                        x, y, _ = point
                        point = [x, y, 1]
                    break
            points = np.vstack([points, point])
            drones = np.hstack((drones, Drone(point, self.body.orientation, drone=pioner)))
        points = points[1:]
        self.drones = drones
        self.body.point = set_barycenter(points)

    def fasten(self) -> None:
        """
        Данная функция перемещает дроны в установленный скелет массива
        :return: None
        """
        if self.apply:
            for drone in self.drones:
                drone.goto(drone.body.point, drone.body.orientation, apply=True)

    def unfasten(self) -> None:
        """
        Функция раскрепляет дроны друг от друга
        :return: None
        """
        pass

    def apply_position(self) -> None:
        """
        Данная функция отправляет дронам изменившеюся ориентацию и позицию в orientation и в point
        :return: None
        """
        old_orinetation = self.body.orientation
        for drone in self.drones:
            drone.apply_position(old_orinetation)

    def euler_rotate(self, alpha: float, beta: float, gamma: float, apply: bool = False) -> None:
        self.rot_z(alpha)
        self.rot_x(beta)
        self.rot_z(gamma)
        if self.apply & apply:
            self.apply_position()

    def rot_x(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = False) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси x. Положительным вращением
        считается по часовой стрелке при направлении оси к нам.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        for drone in self.drones:
            drone.rot_x(angle, rot_point, apply)
        self.trans(rot_x, angle=angle, rot_point=rot_point, apply=apply)

    def rot_y(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = False) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси y. Положительным вращением
         считается по часовой стрелке при направлении оси к нам.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        for drone in self.drones:
            drone.rot_y(angle, rot_point, apply)
        self.trans(rot_y, angle=angle, rot_point=rot_point, apply=apply)

    def rot_z(self, angle: float | int, rot_point: np.ndarray or list = None, apply: bool = False) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси z. Положительным вращением
         считается по часовой стрелке при направлении оси к нам.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        if rot_point is None:
            rot_point = self.body.point
        elif isinstance(rot_point, list):
            rot_point = np.array(rot_point)
        for drone in self.drones:
            drone.rot_z(angle, rot_point, apply=apply)
        self.trans(rot_z, angle=angle, rot_point=rot_point, apply=apply)

    def rot_v(self, angle: float | int,
              rot_point: np.ndarray = np.array([0, 0, 0]),
              axis: list | np.ndarray = np.array([0, 0, 1]),
              apply: bool = False) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси axis, представленной в виде
        вектора. Положительным вращением считается
        по часовой стрелке при направлении оси к нам.
        :param axis:
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        for drone in self.drones:
            drone.rot_v(angle, rot_point, axis=axis)
        new_xyz = rot_v(self.body.point - rot_point, angle, axis=axis) + rot_point

        self.body.orientation = rot_v(self.body.orientation, angle, axis=axis)
        self.trajectory_write(self.body.point, new_xyz)
        self.body.point = new_xyz
        if self.apply & apply:
            self.apply_position()

    def trans(self, func, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
        """
        Данная функция преобразует координаты и ориентацию массива дронов с помощью функции преобразования func. При
        этом положение дронов данная функция не меняет.
        :param func: Функция преобразования для ориентации и точки массива
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Координаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.body.orientation = func(self.body.orientation, angle)
        new_xyz = func(self.body.point - rot_point, angle) + rot_point
        self.trajectory_write(self.body.point, new_xyz)
        self.body.point = new_xyz
        if self.apply & apply:
            self.apply_position()

    def goto(self, point: list | np.ndarray | None = None, orientation: list | np.ndarray | None = None) -> None:
        """
        Функция перемещает Массив дронов в заданное положение. Если задана точка назначения и вектор ориентации, то
        массив дронов поменяет полную ориентацию в пространстве. Если задана только ориентация или только точка, то
        изменится только нужный параметр. Если не задано ничего, то ничего не поменяется.
        :param orientation: Ориентация, представляющая собой матрицу 3x3 из единичных векторов-строк. Первый вектор
         задает x, второй y, третий z.
        :type orientation: list or np.ndarray or None
        :param point: Точка назначения для дрона
        :type point: list[float, int] or None
        :return: None
        """

        if point is None and orientation is None:
            return
        if isinstance(point, list):
            point = np.array(point)
        if isinstance(orientation, list):
            orientation = np.array(orientation)
        for drone in self.drones:
            # Данная формула появилась благодаря векторным операциям
            drone.body.point = point - self.body.point + drone.body.point

        if orientation is not None:
            # Здесь будет функция смены отображения ориентации, как self.trajectory_write()
            self.body.orientation = orientation

        if point is not None:
            self.trajectory_write(self.body.point, point)
            self.body.point = point

        if self.apply:
            self.apply_position()

    def go_traj(self, traj: Curve) -> None:
        for point in traj:
            self.goto(point)
            t = threading.Thread(target=self.wait_for_point)
            t.start()
            while self.occupancy:
                pass

    def wait_for_point(self):
        import time
        self.occupancy = True
        eq = np.zeros((np.shape(self.drones)))
        while self.occupancy:
            array = self.drones
            # eq = np.zeros((np.shape(self.drones)))
            for i, drone in enumerate(self.drones):
                if drone.drone.point_reached():
                    eq[i] = True
                    np.delete(array, i)
            loguru.logger.debug(self.drones[0].drone.point_reached())
            loguru.logger.debug(eq)
            if np.all(eq == True):
                self.occupancy = False
                break
            time.sleep(0.5)

        # th.current_thread()

    def trajectory_write(self, previous_xyz: list | ndarray, current_xyz: list | ndarray) -> None:
        """
        Функция сохраняет траекторию движения массива дронов по точкам во внутренний массив с объектами класса
        Line_segment
        :param previous_xyz: Начальная точка траектории
        :type previous_xyz: list | np.ndarray
        :param current_xyz: Текущая точка траектории
        :type current_xyz:  list | np.ndarray
        :return: None
        """
        segment = Line_segment(point1=previous_xyz, point2=current_xyz)
        segment.color = 'red'
        self.trajectory = np.hstack((self.trajectory, segment))

    def show(self, ax) -> None:
        """
        Функция отображает массив дронов на трехмерном графике, принимая внешний ax от matplotlib
        :param ax: Объект сцены matplotlib
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        for drone in self.drones:
            drone.show(ax)
        self.body.show(ax)

    def self_show(self) -> None:
        """
        Функция отображает дрон на трехмерном графике
        :return: None
        """
        dp = Dspl(self.drones, create_subplot=True, qt=True)
        for drone in self.drones:
            drone.show(dp.ax)
        dp.show()

    def show_trajectory(self, ax) -> None:
        """
        Функция отображает траекторию массива дронов на трехмерном графике, принимая внешний ax от matplotlib
        :param ax: Объект сцены matplotlib
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        for drone in self.drones:
            for segment in drone.trajectory:
                segment.show(ax)
            drone.show(ax)

    def arm(self) -> None:
        """
        Функция отправляет команду на включение двигателей на массив дронов
        :return: None
        """
        if self.apply:
            for drone in self.drones:
                drone.arm()

    def takeoff(self) -> None:
        """
        Функция отправляет команду на взлет всех дронов в массиве
        :return: None
        """
        if self.apply:
            for drone in self.drones:
                drone.takeoff()

    def get_polygon(self) -> ndarray[Any, dtype[Any]]:
        """
        Функция возвращает границы каждого дрона в виде объекта многоугольника Polygon
        :return:
        """
        polygons = np.array([])
        for drone in self.drones:
            polygons = np.hstack((polygons, drone.get_polygon()))
        return polygons

    def info(self):
        out_info = ""
        for drone in self.drones:
            out_info += drone.info() + "\n"
        return out_info


class Map:
    def __init__(self, objects: list | np.ndarray, z: int | float = 1):
        self.objects = np.array(objects)
        self.borders = np.array([])
        self.grab_borders()
        self.z = z

    def grab_borders(self):
        for obj in self.objects:
            bord = obj.get_polygon()
            self.borders = np.hstack([self.borders, bord])

    def get_border_without_border_drone(self, drone: Drone) -> ndarray:
        """

        :param drone:
        :return:
        """
        borders = np.array([])
        for obj in self.objects:
            # loguru.logger.debug(obj.__class__)
            # loguru.logger.debug(type(obj) == Darray)
            if type(obj) == Drone:
                # loguru.logger.debug("1")
                if obj is not drone:
                    # loguru.logger.debug("2")
                    bord = obj.get_polygon()
                    borders = np.hstack([borders, bord])
            elif type(obj) == Darray:
                # loguru.logger.debug("4")
                for dr in obj:
                    loguru.logger.debug(dr is obj)
                    if dr is not drone:
                        loguru.logger.debug("5")

                        bord = dr.get_polygon()
                        loguru.logger.debug(f"bord -> {bord}")
                        borders = np.hstack([borders, bord])
        # loguru.logger.debug(f"5 -> {borders}")
        return borders
    # @property
    # def borders(self):
    #     return self.borders

    # @borders.setter
    # def borders(self, borders):
    #     self.borders = borders

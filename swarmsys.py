import ThreeDTool
import numpy as np
from numpy import cos, sin, pi, ndarray
from body import Body
from dspl import Dspl
from ThreeDTool import Line_segment, Polygon


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
    rotate_z = np.array([[cos(angle), -sin(angle), 0],
                         [sin(angle), cos(angle), 0],
                         [0, 0, 1]])
    rot_vector = vector.dot(rotate_z)
    return rot_vector


def rot_v(axis: list | np.ndarray, vector: list | np.ndarray, angle: float | int) -> np.ndarray:
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
    x, y, z = axis
    rotate = np.array([[cos(angle) + (1 - cos(angle)) * x ** 2, (1 - cos(angle) * x * y) -
                        sin(angle) * z, (1 - cos(angle)) * x * z + sin(angle) * y],
                       [(1 - cos(angle)) * y * x + sin(angle) * z, cos(angle) +
                        (1 - cos(angle)) * y ** 2, (1 - cos(angle)) * y * z - sin(angle) * x],
                       [(1 - cos(angle)) * z * x - sin(angle) * y, (1 - cos(angle)) * z * y +
                        sin(angle) * x, cos(angle) + (1 - cos(angle)) * z ** 2]])
    rot_vector = rotate.dot(vector)
    return rot_vector


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
                 drone=None):
        """
        В данной реализации дрон имеет координату point и вектор ориентации в глобальной системе координат
        :param point:
        :param orientation: Ориентация, представляющая собой матрицу 3x3 из единичных векторов-строк. Первый вектор
        задает x, второй y, третий z.
        """
        self.body = Body(point, orientation)
        self.trajectory = np.array([])
        self.drone = drone
        self.apply = True
        self.begin_point = point
        # характеристики дрона
        self.height = 0.12
        self.length = 0.29
        self.width = 0.29
        self.rad = np.linalg.norm([self.length / 2, self.width / 2])

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

    def goto(self, point: list | np.ndarray | None = None, orientation: list | np.ndarray | None = None) -> None:
        """
        Функция перемещает дрон в заданное положение. Если задана точка назначения и вектор ориентации, тогда
        изменится все. Задана только ориентация или только точка, то изменится только нужный параметр.
        Если не задано ничего, то ничего не поменяется.
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
        if orientation is not None:
            # Здесь будет функция смены отображения ориентации, как self.trajectory_write()
            self.body.orientation = orientation
        if point is not None:
            self.trajectory_write(self.body.point, point)
            self.body.point = point
        if self.apply:
            self.apply_position()

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

    def apply_position(self) -> None:
        """
        Данная функция отправляет дронам изменившеюся ориентацию и позицию в orientation и в point
        :return: None
        """
        if self.drone is not None:
            self.drone.go_to_local_point(self.body.point[0],
                                         self.body.point[1],
                                         self.body.point[2],
                                         yaw=ThreeDTool.angle_from_vectors(np.array([1, 0, 0]),
                                                                           self.body.orientation[0]))

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
        self.rot_z(alpha)
        self.rot_x(beta)
        self.rot_z(gamma)
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

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
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

    def trans(self, func, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
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
        self.body.orientation = func(self.body.orientation, -angle)
        new_point = func(self.body.point - rot_point, angle) + rot_point
        self.trajectory_write(self.body.point, new_point)
        self.body.point = new_point
        if self.apply & apply:
            self.apply_position()

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
        from ThreeDTool.points import Points
        points = Points(self.trajectory, method="plot")
        points.show(ax)

    def arm(self) -> None:
        """
        Функция отправляет команду на включение двигателей на дрон
        :return: None
        """
        if self.apply:
            self.drone.arm()

    def takeoff(self) -> None:
        """
        Функция отправляет команду на взлет дрона
        :return: None
        """
        if self.apply:
            self.drone.takeoff()


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
                 axis_length: int | None = 5):
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
        self.drones = drones
        self.apply = apply
        self.trajectory = np.array([])
        self.body = Body(xyz, orientation)
        self.body.length = axis_length

    def __getitem__(self, item):
        return self.drones[item]

    def create_square_array(self, sizes: list | np.ndarray = np.array([[-10, 10], [-10, 10]]),
                            number_of_drones: int = 4,
                            center_point: np.ndarray = np.array([0, 0, 0]),
                            pio_drones=None) -> None:
        """
        Функция генерирует квадратный массив из дронов
        :param sizes: Размер массива по x и по y
        :type sizes: list or np.ndarray
        :param number_of_drones: Количество дронов
        :type number_of_drones: int
        :param center_point: Центральная точка формирования массива дронов
        :type center_point: np.ndarray
        :param pio_drones: Входящие объекты дронов
        :type pio_drones: list | np.ndarray
        :return: None
        """
        self.body.point = center_point
        x = np.linspace(sizes[0, 0], sizes[0, 1], number_of_drones) + center_point[0]
        y = np.linspace(sizes[1, 0], sizes[1, 1], number_of_drones) + center_point[1]
        z = np.ones(number_of_drones ** 2)
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

    def fasten(self) -> None:
        """
        Данная функция перемещает дроны в установленный скелет массива
        :return: None
        """
        if self.apply:
            for drone in self.drones:
                drone.goto(drone.body.point, drone.body.orientation)

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
        for drone in self.drones:
            drone.apply_position()

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

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = False) -> None:
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
        for drone in self.drones:
            drone.rot_z(angle, rot_point, apply)
        self.trans(rot_z, angle=angle, rot_point=rot_point, apply=apply)

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

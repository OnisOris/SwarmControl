import numpy as np


class Drone:
    def __init__(self, point: list = [0, 0, 0] | np.ndarray[0, 0, 0],
                 orientation: list = [0, 0, 0] | np.ndarray[0, 0, 0]):
        """
        В данной реализации дрон имеет координату point и вектор ориантации orintation в глобальной системе координат
        :param point:
        :param orientation:
        """
        self.point = point
        self.orientation = orientation

    def goto(self, point: None = None | list | np.ndarray, orientation: None = None | list | np.ndarray):
        """
        Функция перемещает дрон в заданное положение
        :param point:
        :return:
        """

        # Здесь будет код для перемещения дрона, например через piosdk

        self.orientation = orientation
        self.point = point


class Darray:
    """
    Данный класс хранит в себе массив с дронами и имеет методы для их общего управления
    """

    def __init__(self, drones: list[Drone] | np.ndarray[Drone],
                 xyz=list[0, 0, 0] | np.ndarray[0, 0, 0],
                 orientation=np.ndarray[1, 0, 0]):
        self.drones = drones
        self.xyz = xyz  # Координата нуля локальной системы координат
        self.orientation = orientation  # Вектор ориентации системы коорднат массива дронов

    def __getitem__(self, item):
        return self.drones[item]

    def create_square_array(self, count: int) -> None:
        pass

    def fasten(self) -> None:
        """
        Данная функция перемещает дроны в установленный скелет массива
        :return:
        """
        pass

    def unfasten(self) -> None:
        """
        Функция раскрепляет дроны друг от друга
        :return:
        """
        pass

    def rotX(self, angle: float | int) -> None:
        rotateX = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
        self.orientation = rotateX.dot(self.orientation)

    def rotY(self, angle: float | int) -> None:
        rotateY = np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]])
        self.orientation = rotateY.dot(self.orientation)
    def rotZ(self, angle: float | int) -> None:
        rotateZ = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
        self.orientation = rotateZ.dot(self.orientation)

import numpy as np


def tr_cre():
    pass


class Map:
    def __init__(self, objects: list | np.ndarray):
        self.objects = objects
        self.borders = np.array([])

    def grab_borders(self):
        for obj in self.objects:
            bord = obj.get_polygon()
            self.borders = np.hstack([self.borders, bord])


class Sheduler:
    """
    Данный класс принимает в себя ссылку на карту с объектами классов: Drone, Darray.
    """
    def __init__(self, map_object: Map):
        pass

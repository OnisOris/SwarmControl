import numpy as np

class Map:
    def __init__(self, objects: list | np.ndarray):
        self.objects = objects
        self.borders = np.array([])

    def grab_borders(self):
        for obj in self.objects:
            bord = obj.get_border()
            self.borders = np.hstack([self.borders, bord])



class Sheduler:
    def __init__(self, map_object: Map):
        pass

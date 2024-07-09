import numpy as np

from vector0 import Vector
class Point:
    def __init__(self, coords, mass=1.0, q=1.0, speed=None, **properties):
        self.coords = np.array(coords)
        if speed is None:
            self.speed = Vector(*[0 for i in range(len(coords))])
        else:
            self.speed = np.array(speed)
        self.acc = Vector(*[0 for i in range(len(coords))])
        self.mass = mass
        self.__params__ = ["coords", "speed", "acc", "q"] + list(properties.keys())
        self.q = q
        for prop in properties:
            setattr(self, prop, properties[prop])

    def set_speed(self, speed: list | np.ndarray | Vector):
        self.speed = Vector(speed)

    def move(self, dt):
        self.coords = self.coords + self.speed * dt

    def accelerate(self, dt):
        self.speed = self.speed + self.acc * dt

    def accinc(self, force):  # Зная сообщаемую силу мы получаем нужное ускорение
        self.acc = self.acc + force / self.mass

    def clean_acc(self):
        self.acc = self.acc * 0
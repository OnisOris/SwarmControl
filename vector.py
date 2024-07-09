import numpy as np

class Vector(np.ndarray):
    def __new__(cls, el):
        super().__new__(cls, el)
    def __init__(self, *el):
        pass
        # for e in el:
        #     self = np.hstack([self, e])


    # def hstack(self, el):
    #     return



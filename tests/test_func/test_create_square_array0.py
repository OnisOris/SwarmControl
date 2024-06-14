import numpy as np
from icecream import icecream as ic
from numpy import pi
from swarmsys import *


def test_create_square_array():
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
    # d = np.allclose(test_matrix, eq_matrix, 1e-8)
    assert np.allclose(test_matrix, eq_matrix, 1e-8)

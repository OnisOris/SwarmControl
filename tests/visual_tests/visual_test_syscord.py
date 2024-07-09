from syscord import Syscord
from point import Point
import numpy as np

point = Point([-10, 0, 0], speed=np.array([100, 0, 0]))
point2 = Point([3, 3, 0], speed=np.array([100, 0, 0]))
sys = Syscord([point, point2])

sys.real_time_start_sim()

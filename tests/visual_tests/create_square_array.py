from icecream import icecream as ic
from numpy import pi
from swarmsys import *
from ThreeDTool import Points
# import pytest
rot_point = np.array([10, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)

arr = Darray()
arr.create_square_array(number_of_drones=2, center_point=np.array([10, 0, 0]))
arr.rot_z(-pi/2, rot_point, apply=True)
dp = Dspl([arr] + [p], qt=True)
lim_1 = -15
lim_2 = 15
dp.limits(x=[lim_1+10, lim_2+10], y=[lim_1, lim_2], z=[lim_1, lim_2])
dp.ax.title.set_text('Создание массива дронов')
dp.show()
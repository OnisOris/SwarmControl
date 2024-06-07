from icecream import icecream
from numpy import pi
from swarmsys import *
from ThreeDTool import Points
rot_point = np.array([10, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)
# arrs = []
# for i in range(8):
#     arrs.append(Darray())
#     for arr in arrs:
#         arr.create_square_array(center_point=np.array([10, 0, 0]))
#         arr.rot_z(np.pi/4*i, rot_point=rot_point, apply=True)

arr = Darray()
icecream.ic()
arr.create_square_array(center_point=np.array([10, 0, 0]))
arr.rot_z(-pi/2, rot_point, apply=True)

# dp = Dspl(arrs + [p], qt=True)
dp = Dspl([arr] + [p], qt=True)
dp.show()
# arr.self_show()

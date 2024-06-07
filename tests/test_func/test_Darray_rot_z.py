from swarmsys import *
from ThreeDTool import Points

rot_point = np.array([40, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)
arrs = []
for i in range(8):
    arrs.append(Darray())
for j, arr in enumerate(arrs):
    arr.create_square_array(center_point=np.array([0, 0, 0]))
    arr.rot_z(np.pi/4*j, rot_point=rot_point, apply=True)

dp = Dspl(arrs + [p], qt=True)
dp.limits(x=[-10, 80], y=[-10, 80], z=[-10, 80])
dp.show()

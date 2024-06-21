from swarmsys import *
from ThreeDTool import Points
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')

rot_point = np.array([40, 0, 0])
zero_point = np.array([0, 0, 0])
axis = np.array([1, 1, 1])
p = Points([rot_point, zero_point], text=True)
arrs = []
for i in range(8):
    arrs.append(Darray())
for j, arr in enumerate(arrs):
    arr.create_square_array(center_point=np.array([0, 0, 0]))
    arr.rot_v(np.pi/4*j, rot_point=rot_point, axis=axis, apply=True)

dp = Dspl(arrs + [p])
lim_1 = -20
lim_2 = 30
dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])
dp.ax.quiver(*rot_point, *axis,
                      length=20, color='orange')
# dp.ax.title.set_text(f'Вращение дрона по оси z и точки {rot_point}')
dp.ax.title.set_text(f'Вращение массива дронов по оси z и точки {rot_point}')
dp.show()

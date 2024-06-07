from body import Body
from ThreeDTool import Dspl
import matplotlib.pyplot as plt

b = Body()
d = Dspl([b], qt=True)
d.ax.set_xlim([-1, 1])
d.ax.set_ylim([-1, 1])
d.ax.set_zlim([-1, 1])
plt.axis()
d.show()

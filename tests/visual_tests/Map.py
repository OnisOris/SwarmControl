from scheduler import *
from swarmsys import *
from ThreeDTool import Dspl, Line_segment, Points
import matplotlib
matplotlib.use('WebAgg')

darr = Darray()
darr.create_square_array(sizes=np.array([[-1, 1],
                                         [-1, 1]]),
                         number_of_drones=2)

m = Map([darr])

# m.grab_borders()

darr[0].calculate_path([1, -1, 1], m)

# ls = np.array([])
# points = []
# for i, drone in enumerate(darr):
#     if i != np.shape(darr.drones)[0]-1:
#         ls = np.hstack((ls, Line_segment(point1=drone.body.point, point2=darr[i+1].body.point)))
#         points.append(drone.get_polygon().point_of_intersection(darr[i+1].body.point))
#     else:
#         ls = np.hstack((ls, Line_segment(point1=drone.body.point, point2=darr[0].body.point)))
#         points.append(drone.get_polygon().point_of_intersection(darr[0].body.point))


# pp = []
# for p in points:
#     pp.append(Points(p, s=15, color='red'))

# dp = Dspl(np.hstack([m.borders, ls, pp]))
# dp = Dspl(pp)
# dp.show()


from SwarmControl.swarmsys import *
import numpy as np
# from scheduler import Map
from loguru import logger

darr = Darray()
darr.create_square_array(sizes=np.array([[-1, 1],
                                         [-1, 1]]),
                         number_of_drones=2)

m = Map([darr])

m.grab_borders()

logger.debug(darr.info())

# darr[1].calculate_path([-2, -2, 1], map_object=m)
#Здесь всегда правда
print(darr[1].check_collision(darr[2], map_object=m))
# TODO: сделать так, чтобы происходила проверка точек орезков границ дрона

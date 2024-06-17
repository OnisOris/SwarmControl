from ThreeDTool import loxodrome
from swarmsys import Drone
import numpy as np
from numpy import cos, sin


class TestDrone:
    def test_goto_drone(self):
        drone = Drone()
        traj = loxodrome(R=10, step=1, point_n=np.array([0, 0, 10]))
        for point in traj:
            drone.goto(point)
        assert np.allclose(drone.body.point, traj[-1], 1e-8)

    def test_Drone_rot_x(self):
        rot_point = np.array([40, 40, 0])
        point = np.array([0, 0, 0])
        drone = Drone(point=point, orientation=np.array([[1, 0, 0],
                                                         [0, 1, 0],
                                                         [0, 0, 1]]))
        angle = np.pi / 4
        drone.rot_x(angle, rot_point=rot_point, apply=True)
        rotate_x = np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        assert np.allclose(drone.body.point,
                           (point - rot_point).dot(rotate_x) + rot_point,
                           1e-8)

    def test_Drone_rot_y(self):
        rot_point = np.array([40, 40, 0])
        point = np.array([0, 0, 0])
        drone = Drone(point=point, orientation=np.array([[1, 0, 0],
                                                         [0, 1, 0],
                                                         [0, 0, 1]]))
        angle = np.pi / 4
        drone.rot_y(angle, rot_point=rot_point, apply=True)
        rotate_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        assert np.allclose(drone.body.point,
                           (point - rot_point).dot(rotate_y) + rot_point,
                           1e-8)

    def test_Drone_rot_z(self):
        rot_point = np.array([40, 40, 0])
        point = np.array([0, 0, 0])
        drone = Drone(point=point, orientation=np.array([[1, 0, 0],
                                                         [0, 1, 0],
                                                         [0, 0, 1]]))
        angle = np.pi / 4
        drone.rot_z(angle, rot_point=rot_point, apply=True)
        rotate_z = np.array([[cos(angle), -sin(angle), 0],
                             [sin(angle), cos(angle), 0],
                             [0, 0, 1]])
        assert np.allclose(drone.body.point,
                           (point - rot_point).dot(rotate_z) + rot_point,
                           1e-8)

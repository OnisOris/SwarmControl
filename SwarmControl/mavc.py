import numpy as np
from pymavlink import mavutil
import time
import threading


def _create_connection(connection_method, ip, port):
    """
    create mavlink connection
    :return: mav_socket
    """
    mav_socket = mavutil.mavlink_connection('%s:%s:%s' % (connection_method, ip, port))
    return mav_socket


class Mavc:
    def __init__(self, ip='10.1.100.114', mavlink_port: int = 8001, connection_method: str = 'udpout'):
        self.mavlink_socket = _create_connection(connection_method=connection_method,
                                                 ip=ip, port=mavlink_port)
        self._heartbeat_timeout = 1
        self._mavlink_send_number = 10
        self._heartbeat_send_time = time.time() - self._heartbeat_timeout
        # Flag for the concurrent thread. Signals whether or not the thread should go on running
        self.__is_socket_open = threading.Event()
        self.__is_socket_open.set()
        self._message_handler_thread = threading.Thread(target=self._message_handler, daemon=True)
        self._message_handler_thread.daemon = True
        self._message_handler_thread.start()
        self.attitude = np.array([0, 0, 0, 0, 0, 0])

    def arm(self):
        return self._send_command_long(command_name='ARM', command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                       param1=1)

    def disarm(self):
        return self._send_command_long(command_name='DISARM', command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                       param1=0)

    def takeoff(self):
        return self._send_command_long(command_name='TAKEOFF', command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)

    def land(self):
        return self._send_command_long(command_name='LAND', command=mavutil.mavlink.MAV_CMD_NAV_LAND)

    def get_position(self):
        """
        Функция вернет ndarray (6,) с координатами x, y, z, vx, vy, vz
        :return: np.ndarray
        """
        return self.attitude

    def go_to_local_point(self, x, y, z, yaw):
        """ Flight to point in the current navigation system's coordinate frame """
        cmd_name = 'GO_TO_POINT'
        mask = 0b0000_10_0_111_111_000  # _ _ _ _ yaw_rate yaw   force_set   afz afy afx   vz vy vx   z y x
        x, y, z = y, x, -z  # ENU coordinates to NED coordinates
        return self._send_position_target_local_ned(command_name=cmd_name,
                                                    coordinate_system=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                                    mask=mask, x=x, y=y, z=z, yaw=yaw)

    def set_manual_speed(self, vx, vy, vz, yaw_rate):
        """ Set manual speed """
        cmd_name = 'MANUAL_SPEED'
        mask = 0b0000_01_0_111_000_111  # _ _ _ _ yaw_rate yaw   force_set   afz afy afx   vz vy vx   z y x
        vx, vy, vz = vy, vx, -vz  # ENU coordinates to NED coordinates
        return self._send_position_target_local_ned(command_name=cmd_name,
                                                    coordinate_system=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                                    mask=mask, vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def _send_position_target_local_ned(self, command_name, coordinate_system, mask=0b0000_11_0_111_111_111, x=0, y=0,
                                        z=0, vx=0, vy=0, vz=0, afx=0, afy=0, afz=0, yaw=0, yaw_rate=0,
                                        target_system=None, target_component=None):
        print(f"_send_position_target_local_ned --> {command_name}")
        if target_system is None:
            target_system = self.mavlink_socket.target_system
        if target_component is None:
            target_component = self.mavlink_socket.target_component
        for confirm in range(self._mavlink_send_number):
            self.mavlink_socket.mav.set_position_target_local_ned_send(0, target_system, target_component,
                                                                       coordinate_system,
                                                                       mask, x, y, z, vx, vy, vz, afx, afy, afz,
                                                                       yaw, yaw_rate)
        return False

    def _send_command_long(self, command_name, command, param1: float = 0, param2: float = 0, param3: float = 0,
                           param4: float = 0, param5: float = 0, param6: float = 0, param7: float = 0,
                           target_system=None, target_component=None):
        print(f"_send_command_long --> {command_name}")
        if target_system is None:
            target_system = self.mavlink_socket.target_system
        if target_component is None:
            target_component = self.mavlink_socket.target_component
        confirm = 0
        while True:
            self.mavlink_socket.mav.command_long_send(target_system, target_component, command, confirm,
                                                      param1, param2, param3, param4, param5, param6, param7)
            confirm += 1
            if confirm >= self._mavlink_send_number:
                break

    def _send_heartbeat(self):
        self.mavlink_socket.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                               mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        self._heartbeat_send_time = time.time()

    def _message_handler(self):
        while True:
            if not self.__is_socket_open.is_set():
                break

            if time.time() - self._heartbeat_send_time >= self._heartbeat_timeout:
                self._send_heartbeat()
            msg = self.mavlink_socket.recv_msg()
            if msg is not None:
                # LOCAL_POSITION_NED {time_boot_ms : 8700899,
                # x : 3.2046821117401123,
                # y : -1.0340602397918701,
                # z : 0.039611928164958954,
                # vx : -0.0025489432737231255,
                # vy : 0.0005856414791196585,
                # vz : -0.022763948887586594}
                if msg.get_type() == "LOCAL_POSITION_NED":
                    self.attitude = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz])

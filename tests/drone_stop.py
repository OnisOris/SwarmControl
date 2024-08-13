from pioneer_sdk import Pioneer
drone = Pioneer(ip="10.1.100.104", mavlink_port=5656)
drone.land()
drone.disarm()
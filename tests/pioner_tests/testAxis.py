from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
from config import CONFIG

num = CONFIG['standard_port']
drone = Pioneer(ip=f"{CONFIG['ip_3']}{CONFIG['num_drone']}", mavlink_port=CONFIG['standard_port'])
dr = Drone(CONFIG, drone=drone)

dr.arm()
dr.takeoff()
dr.wait_point = True
dr.goto([-3, -3, 1.5], apply=True)
dr.goto([3, 3, 1.5], apply=True)




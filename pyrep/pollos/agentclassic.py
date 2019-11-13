import numpy as np
import time, math
from pynput.keyboard import Key, Listener
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.dummy import Dummy
from environment import EnvPollos

class Agent(object):
    state = "WAIT_FOR_CHICKEN"
    reloj = time.time()
    epochs = 0
   
    # simple state machine
    def act(self, env):

        env.pr.step()
        self.np_pollo_target = np.array(env.pollo_target.get_position())
        self.np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        self.np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
        self.dist = np.linalg.norm(self.np_robot_tip_position - self.np_pollo_target)
        
        # state machine
        if self.state == "WAIT_FOR_CHICKEN":
            return(self.wait_for_chicken(env))
        elif self.state == "HANGUP":
            return(self.hangup(env))
        elif self.state == "INIT_GRAB":
            return(self.init_grab(env))
        elif self.state == "GRAB":
            return(self.grab(env))
        elif self.state == "WAIT":
            return(self.wait(env))
        elif self.state == "RESET_EPISODE":
            return(self.resetEpisode(env))
        elif self.state == "IDLE":
            pass

    def wait_for_chicken(self, env):
        if time.time() - self.reloj > 5:
            self.state = "RESET_EPISODE"
        if self.dist < 0.30:
            self.state = "INIT_GRAB"
        else:
            local_path = CartesianPath.create(show_line = True, show_orientation = True,
                                            show_position = True, automatic_orientation = True)   # first point is lasy to execute
            np_local_tip = self.np_robot_tip_position
            np_local_tip[0] = self.np_pollo_target[0]
            #local_path.insert_control_points([list(np_local_tip) + list(self.np_robot_tip_orientation)])
            local_path.insert_control_points([list(np_local_tip) + env.pollo_target.get_orientation()])
            local_path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
            local_ang_path = env.agent.get_path_from_cartesian_path(local_path)
            while not local_ang_path.step():
                env.pr.step()
            #angles = env.agent.get_joint_positions()
            #env.agent.joints[4].set_joint_target_position(env.pollo_target.get_orientation()[1])
            local_path.remove()
    
    def init_grab(self, env):
        self.path = CartesianPath.create(show_line = True, show_orientation = True,
                                    show_position = True, automatic_orientation = False, 
                                    keep_x_up = False)   # first point is lasy to execute
        np_elevated_final_point = np.add(self.np_pollo_target, np.array([0.0,0.0,0.30]))
        self.path.insert_control_points([list(np_elevated_final_point) + list(self.np_robot_tip_orientation)])
        np_after_pollo = np.add(self.np_pollo_target, np.array([0.0,0.0, 0.06]))
        self.path.insert_control_points([list(np_after_pollo) + list(self.np_robot_tip_orientation)])
        np_predicted_pollo = np.add(self.np_pollo_target, np.array([0.0,-0.10,0.1]))
        self.path.insert_control_points([list(np_predicted_pollo) + list(self.np_robot_tip_orientation)])
        self.path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
        
        try:
            self.ang_path = env.agent.get_path_from_cartesian_path(self.path)
            self.state = "GRAB"
        except IKError as e:
            print('Agent::grasp    Could not find joint values')   
            self.state = "RESET_EPISODE"

    def grab(self, env):
        if self.ang_path.step():
            angles = env.agent.get_joint_positions()
            angles[5] = angles[5] - 0.5
            angles[4] = 0.0
            env.agent.joints[4].set_joint_target_position(angles[4])
            env.agent.joints[5].set_joint_target_position(angles[5])
            
            self.reloj = time.time()
            self.state = "HANGUP"
            self.path.remove()
            
    def hangup(self, env):
        local_path = CartesianPath.create(show_line = True, show_orientation = True,
                                          show_position = True, automatic_orientation = False)                    
        local_path.insert_control_points([env.waypoints[0].get_position() + env.waypoints[0].get_orientation()])
        local_path.insert_control_points([env.waypoints[1].get_position() + env.waypoints[1].get_orientation()])             
        local_path.insert_control_points([env.waypoints[2].get_position() + env.waypoints[2].get_orientation()])       
        local_path.insert_control_points([env.waypoints[3].get_position() + env.waypoints[3].get_orientation()])       
                       
        local_path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
        local_ang_path = env.agent.get_path_from_cartesian_path(local_path)
        local_ang_path.visualize()  # Let's see what the path looks like
        print('Executing plan ...')
        while not local_ang_path.step():
            env.pr.step()
        local_path.remove()
        # path = env.agent.get_path(position=env.waypoints[0].get_position(),
        #                           quaternion=env.waypoints[0].get_quaternion())
        local_ang_path.clear_visualization()
        self.state = "WAIT"
    
    def wait(self, env):
        if time.time() - self.reloj > 5:
            self.state = "RESET_EPISODE"

    def resetEpisode(self, env):
        self.epochs += 1
        print("Resetting environment:", self.epochs, " epochs")
        env.reset()
        self.reloj = time.time()
        self.state = "WAIT_FOR_CHICKEN"
       
    ########################3

env = EnvPollos()
agent = Agent()

try:
    while True:
        agent.act(env)
        #time.sleep(0.050)
except KeyboardInterrupt:
    pass

env.close()
        
    
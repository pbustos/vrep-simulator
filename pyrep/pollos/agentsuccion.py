import numpy as np
import time, math
import keyboard
from pyrep.objects.cartesian_path import CartesianPath
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
        elif self.state == "INIT_GRAB":
            return(self.init_grab(env))
        elif self.state == "GRAB":
            return(self.grab(env))
        elif self.state == "LEVANTAR":
            return(self.levantar(env))
        elif self.state == "DEJAR":
            return(self.dejar(env))    
        elif self.state == "WAIT":
            return(self.wait(env))
        elif self.state == "RESET_EPISODE":
            return(self.resetEpisode(env))
    
    def wait_for_chicken(self, env):
        if time.time() - self.reloj > 5:
            self.state = "RESET_EPISODE"
        if np.fabs(self.np_pollo_target[0]-self.np_robot_tip_position[0]) < 0.20:
            self.state = "INIT_GRAB"
        else:
            local_path = CartesianPath.create(show_line = True, show_orientation = True,
                                            show_position = True, automatic_orientation = True)   # first point is lasy to execute
            np_local_tip = self.np_robot_tip_position
            np_local_tip[1] = self.np_pollo_target[1]
            local_path.insert_control_points([list(np_local_tip) + list(self.np_robot_tip_orientation)])
            local_path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
            local_ang_path = env.agent.get_path_from_cartesian_path(local_path)
            while not local_ang_path.step():
                pass
            local_path.remove()
    
    def init_grab(self, env):
        self.path = CartesianPath.create(show_line = True, show_orientation = True,
                                         show_position = True, automatic_orientation = True, 
                                         keep_x_up = False)   # first point is lasy to execute
        np_predicted_pollo = np.add(self.np_pollo_target, np.array([-0.18, 0, -0.16]))
        #print(self.np_robot_tip_orientation,  env.initial_agent_tip_euler)
        self.path.insert_control_points([list(np_predicted_pollo) + list(env.initial_agent_tip_euler)])
        self.path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
        env.pr.script_call('activate_suction@suctionPad', env.vrep.sim_scripttype_childscript)
        try:
            self.ang_path = env.agent.get_path_from_cartesian_path(self.path)
            self.state = "GRAB"
        except IKError as e:
            print('Agent::grasp    Could not find joint values')   
            self.state = "RESET_EPISODE"

    def grab(self, env):
        if self.ang_path.step():
            self.reloj = time.time()
            self.state = "LEVANTAR"
            self.path.remove()

    def levantar(self, env):
        time.sleep(0.1)
        local_path = CartesianPath.create(show_line = True, show_orientation = True,
                                          show_position = True, automatic_orientation = True)   # first point is lasy to execute
        local_path.insert_control_points([list(env.initial_agent_tip_position) + list(self.np_robot_tip_orientation)])
        local_path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
        local_ang_path = env.agent.get_path_from_cartesian_path(local_path)
        while not local_ang_path.step():
            pass
        local_path.remove()
        self.state = "DEJAR"

    def dejar(self, env):
        r = env.pr.script_call('deactivate_suction@suctionPad', env.vrep.sim_scripttype_childscript)
        self.state = "WAIT"

    def wait(self, env):
        if time.time() - self.reloj > 3:
            self.state = "RESET_EPISODE"

    def resetEpisode(self, env):
        self.epochs += 1
        print("Resetting environment:", self.epochs, " epochs")
        env.reset()
        self.reloj = time.time()
        self.state = "WAIT_FOR_CHICKEN"
        env.pr.script_call('deactivate_suction@suctionPad', env.vrep.sim_scripttype_childscript)
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
        
    
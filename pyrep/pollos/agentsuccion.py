import numpy as np
import time, math
import keyboard
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.errors import ConfigurationPathError, IKError
from environment import EnvPollos
from joyreader import JoyReader

class Agent(object):
    state = "WAIT_FOR_CHICKEN"
    #state = "TIP_ROTATING"
    reloj = time.time()
    epochs = 0
    
    # simple state machine
    def act(self, env, joy):

        env.pr.step()
        self.np_pollo_target = np.array(env.pollo_target.get_position())
        self.np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        self.np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
        self.dist = np.linalg.norm(self.np_robot_tip_position - self.np_pollo_target)
        
        # state machine
        if self.state == "TIP_ROTATING":
            return(self.tip_rotating(env, joy))
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
    
    def tip_rotating(self, env, joy):
        #print(joy.incs)
        try:
            np_new_or = np.add(np.array(env.initial_agent_tip_euler), np.array(joy.incs))
            local_target = np.array(env.agent.get_tip().get_position()) + np.array(joy.incs)
            incs = list((np.array(joy.incs) * 10) + np.array(env.initial_agent_tip_euler))
            angles = env.agent.solve_ik(position=env.initial_agent_tip_position, euler=incs)
            env.agent.set_joint_target_positions(angles)
            env.target.set_orientation(incs)
            #print(angles)
        except IKError as e:
            print('Agent::act    Could not find joint values', e)   

    def wait_for_chicken(self, env):
        #print("WAIT_FOR_CHICKEN")
        if time.time() - self.reloj > 6:
            self.state = "RESET_EPISODE"
        if np.fabs(self.np_pollo_target[0]-self.np_robot_tip_position[0]) < 0.30:
            self.state = "INIT_GRAB"
        else:
            env.target.set_position(list(self.np_pollo_target))
            local_path = CartesianPath.create(show_line = True, show_orientation = True,
                                            show_position = True, automatic_orientation = True)   # first point is lasy to execute
            np_local_tip = self.np_robot_tip_position
            np_local_tip[1] = self.np_pollo_target[1]
            local_path.insert_control_points([list(np_local_tip) +env.agent.get_tip().get_orientation()])
            local_path.insert_control_points([list(self.np_robot_tip_position) + env.agent.get_tip().get_orientation()])
            local_ang_path = env.agent.get_path_from_cartesian_path(local_path)
            while not local_ang_path.step():
                env.pr.step()
            local_path.remove()
    
    def init_grab(self, env):
        print("INIT_GRAB")
        self.path = CartesianPath.create(show_line = True, show_orientation = True,
                                         show_position = True, automatic_orientation = False)   # first point is lasy to execute
        np_predicted_pollo = np.add(self.np_pollo_target, np.array([-0.24, 0.05, -0.05]))
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
        print("GRAB")
        env.target.set_position(list(self.np_pollo_target))
        if self.ang_path.step():
            self.state = "LEVANTAR"
            self.path.remove()

    def levantar(self, env):
        print("LEVANTAR")
        local_path = CartesianPath.create(show_line = True, show_orientation = True,
                                          show_position = True, automatic_orientation = True)   # first point is lasy to execute
        np_high_point = np.add(env.initial_agent_tip_position,np.array([0.0,0.0,0.4]))
        local_path.insert_control_points([list(np_high_point) + list(self.np_robot_tip_orientation)])
        local_path.insert_control_points([list(self.np_robot_tip_position) + list(self.np_robot_tip_orientation)])
        local_ang_path = env.agent.get_path_from_cartesian_path(local_path)
        env.target.set_position(list(self.np_pollo_target))
        while not local_ang_path.step():
            pass
        local_path.remove()
        self.reloj = time.time()
        self.state = "DEJAR"

    def dejar(self, env):
        #print("DEJAR")
        if time.time() - self.reloj > 2:
            self.state = "RESET_EPISODE"
            r = env.pr.script_call('deactivate_suction@suctionPad', env.vrep.sim_scripttype_childscript)
            
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
joy = JoyReader()
joy.start()

try:
    while True:
        agent.act(env, joy)
        #time.sleep(0.050)
except KeyboardInterrupt:
    pass

env.close()
        
    
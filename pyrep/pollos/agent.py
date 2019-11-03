import numpy as np
from pyrep.objects.cartesian_path import CartesianPath
import time, math

#import algs.ddpg

class Agent(object):
    state = "CATCH"

    # simple state machine
    def act(self, env, joy):

        # state machine
        if self.state == "JOYSTICK":
            return(self.joystick(env, joy))
        elif self.state == "INIT_GRASP":
            return(self.init_grasp(env))
        elif self.state == "GRASP":
            return(self.grasp(env))
        elif self.state == "UNLOAD":
            return(self.unload(env))
        elif self.state == "WAIT":
            return(self.wait(env))
        elif self.state == "RESET_EPISODE":
            return(self.resetEpisode(joy))
        elif self.state == "CATCH":
            return(self.catch(env, joy))

    def catch(self, env, joy):
        np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        np_pollo_target = np.array(env.pollo_target.get_position())
        dist = np.linalg.norm(np_robot_tip_position - np_pollo_target)
        angles = env.agent.get_joint_positions()
        if dist < 0.4:
            print("changing to INIT_GRASP")
            self.state = "INIT_GRASP"
            return angles

        np_robot_tip_position[0] = np_pollo_target[0]
        angles = env.agent.solve_ik(position=list(np_robot_tip_position), quaternion=env.initial_agent_tip_quaternion)
        return angles
    
    def joystick(self, env, joy):
        if joy.unloading:
            wrist_angle = math.radians(120)
        else:
            wrist_angle = env.initial_joint_positions[5]
        local_target = np.array(env.agent.get_tip().get_position()) + np.array(joy.incs)
        try:
            angles = env.agent.solve_ik(position=list(local_target), quaternion=env.initial_agent_tip_quaternion)
            angles[5] = wrist_angle
        except IKError as e:
            print('Agent::act    Could not find joint values', e)   
        return angles

    def init_grasp(self, env):
        np_pollo_target = np.array(env.pollo_target.get_position())
        np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
        print("init ", np_robot_tip_orientation)
        dist = np.linalg.norm(np_robot_tip_position - np_pollo_target)
        c_path = CartesianPath.create()
        
        # LIFO: goto table
        c_path.insert_control_points([env.table_target.get_position() + list(np_robot_tip_orientation)])
        
        at100 = np.add(np_pollo_target, np.array([0.0,0.0,0.30]))
        c_path.insert_control_points([list(at100) + list(np_robot_tip_orientation)])
        # c_path.insert_control_points([env.table_target.get_position() + list(np_robot_tip_orientation)])
        # np_robot_tip_orientation[0] += 1
        # c_path.insert_control_points([list(np_pollo_target) + list(np_robot_tip_orientation)])
        # np_robot_tip_orientation[0] -= 1
        np_pollo_target[1] -= 0.1
        c_path.insert_control_points([list(np_pollo_target) + list(np_robot_tip_orientation)])
        # at10 = np.add(np.array(c_path.get_pose_on_path(0.2)[0]), np.array([0.0,0.0,0.1]))
        # c_path.insert_control_points([list(at10) + list(np_robot_tip_orientation)])
        c_path.insert_control_points([list(np_robot_tip_position) + list(np_robot_tip_orientation)])
        
        try:
            #angles = env.agent.solve_ik(position=list(np_pollo_target), orientation=np_robot_tip_orientation)
            self.path = env.agent.get_path_from_cartesian_path(c_path)
            print("at init_grasp ")
        except IKError as e:
            print('Agent::grasp    Could not find joint values')   
        self.state = "GRASP"    
        print("changing to GRASP")
        return None
            
    def grasp(self, env):
        np_pollo_target = np.array(env.pollo_target.get_position())
        np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        dist = np.linalg.norm(np_robot_tip_position - np_pollo_target)
        if self.path.step():
            self.state = "UNLOAD"
        return None

    def unload(self, env):
        angles = env.agent.get_joint_positions()
        angles[5] = math.radians(120)
        self.initReloj(4) # secs
        self.state = "WAIT"
        return angles

    def wait(self, env):
        if (time.time()-self.reloj) > self.DELAY:
            self.state = "RESET_EPISODE"
            np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
            print("after ", np_robot_tip_orientation)
        
        return None

    def resetEpisode(self, joy):
        joy.next_ep = True
        self.state = "CATCH"
        return None

    ########################3
        
    def initReloj(self, delay):
        self.reloj = time.time()
        self.DELAY = delay

    def learn(self, replay_buffer):
        del replay_buffer
        pass

    
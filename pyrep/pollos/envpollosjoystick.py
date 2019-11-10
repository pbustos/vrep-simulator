import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.cartesian_path import CartesianPath
import time, math
from envpollos import EnvPollos
#from envpollospanda import EnvPollos

class EnvPollosJoystick(EnvPollos):
    def __init__(self):
        super().__init__()
    
    def reset(self):
        super().reset()
    
    def step(self, action):
        #state, reward, done, info = super().step(action)
        try:
            start = time.time()
            angles = self.agent.solve_ik(position=action, quaternion=self.initial_agent_tip_quaternion)
            #print("ik ", time.time() - start)
            # move the robot and wait to stop
            self.agent.set_joint_target_positions(angles)
            #print("ik2 ", time.time() - start)
            # while True:         # wait for arm to stop
            #     self.pr.step()  # Step the physics simulation
            #     a = self.agent.get_joint_velocities()
            #     if not np.any(np.where( np.fabs(a) < 0.1, False, True )):
            #         break
        except IKError as e:
            print('Agent::act    Could not find joint values', e)   
        return self._get_state(), 0, True, {}
        
    def close(self):
        super().reset()

    def render(self):
        super().reset()

    #########################

    def _get_state(self):
       j = self.agent.get_joint_positions()
       r = np.array([j[0],j[1],j[2],j[3],j[4],j[5]])
       return r
       pass
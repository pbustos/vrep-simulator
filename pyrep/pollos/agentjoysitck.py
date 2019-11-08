import numpy as np
import time, math
from envpollosjoystick import EnvPollosJoystick 
from joyreader import JoyReader

class Agent():
    state = "JOYSTICK"

    # simple state machine
    def act(self, env, joy):

        # state machine
        if self.state == "JOYSTICK":
            return(self.joystick(env, joy))
        elif self.state == "UNLOAD":
            return(self.unload(env))
        elif self.state == "WAIT":
            return(self.wait(env))
        elif self.state == "RESET_EPISODE":
            return(self.resetEpisode(joy))
    
    def joystick(self, env, joy):
        if joy.next_ep:
            print("changing to RESET")
            self.state = "RESET_EPISODE"
        else:
            local_target = np.array(env.agent.get_tip().get_position()) + np.array(joy.incs)
            env.step(list(local_target))

    def unload(self, env):
        # angles = env.agent.get_joint_positions()
        # angles[5] = math.radians(120)
        # self.initReloj(4) # secs
        # self.state = "WAIT"
        # return angles
        pass

    def wait(self, env):
        # if (time.time()-self.reloj) > self.DELAY:
        #     self.state = "RESET_EPISODE"
        #     np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
        #     print("after ", np_robot_tip_orientation)
        # return None
        pass

    def resetEpisode(self, joy):
        print("Resetting environment")
        env.reset()
        joy.next_ep = False
        self.state = "JOYSTICK"
       
    ########################3

env = EnvPollosJoystick()
joy = JoyReader()
agent = Agent()
joy.start()

while not joy.end:    
    agent.act(env, joy)
    #time.sleep(0.01)

env.close()
        
    
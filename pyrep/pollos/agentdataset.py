import numpy as np
import time, math
from pynput.keyboard import Key, Listener
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.dummy import Dummy
from environment import EnvPollos
from matplotlib import pyplot as plt


class Agent(object):
    state = "WATCH_CHICKEN"
    reloj = time.time()
    epochs = 0
    frames = 0
       
    # simple state machine
    def act(self, env):

        env.pr.step()
        self.np_pollo_target = np.array(env.pollo_target.get_position())
        self.np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        self.np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
        self.dist = np.linalg.norm(self.np_robot_tip_position - self.np_pollo_target)
        
        # state machine
        if self.state == "WATCH_CHICKEN":
            return(self.watch_chicken(env))
        elif self.state == "RESET_EPISODE":
            return(self.resetEpisode(env))
        
    def watch_chicken(self, env):
        if self.dist < 0.15 or (time.time() - self.reloj) > 3:
            self.state = "RESET_EPISODE"
        depth = env.camera.capture_depth()
        
        np_pollo_target = np.array(env.pollo_target.get_position())
        np_pollo_en_camara = env.np_camera_matrix_extrinsics.dot(np.append([np_pollo_target],1.0))
        np_pollo_en_camara_img = env.np_camera_matrix_intrinsics.dot(np_pollo_en_camara)
        np_pollo_en_camara_img = np_pollo_en_camara_img / np_pollo_en_camara[2]
        np_pollo_en_camara_img = np.delete(np_pollo_en_camara_img,2)
        res = np.array([depth, np_pollo_en_camara])
        file = f"dataset/pollos_xyz_{self.frames}.npy"
        np.save(file, res)
        self.frames  += 1
        print(self.frames)
        
        circ = plt.Circle((int(np_pollo_en_camara_img[0]), int(np_pollo_en_camara_img[1])),10)
        plt.clf()
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circ)
        circ.set_facecolor(np.array([0,0,0]))
        ax.imshow(depth, cmap = 'hot')
        plt.pause(0.000001)

    def resetEpisode(self, env):
        self.epochs += 1
        print("Resetting environment:", self.epochs, " epochs")
        env.reset()
        self.reloj = time.time()
        self.state = "WATCH_CHICKEN"
        

    ########################################   
    

env = EnvPollos()
agent = Agent()

try:
    while True:
        agent.act(env)
        #time.sleep(0.050)
except KeyboardInterrupt:
    pass

env.close()
        
    
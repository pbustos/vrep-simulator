from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.cartesian_path import CartesianPath
import numpy as np
import math, time
from pyrep.backend import vrep
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

SCENE_FILE = '/home/pbustos/software/vrep-simulator/pollos/pollos.ttt'

class EnvPollos(Env):
    def __init__(self, ep_length=100):
        """
        Pollos environment for testing purposes
        :param dim: (int) the size of the dimensions you want to learn
        :param ep_length: (int) the length of each episodes in timesteps
        """
        # 5 points in 3D space forming the trajectory
        self.action_space = Box(low=-0.3, high=0.3, shape=(5,3), dtype=np.float32)
        self.observation_space = Box(low=np.array([-0.1, -0.2, 0, -1.5, 0.0, 0.0, -0.5, -0.2, -0.2]), 
                                     high=np.array([0.1, 1.7, 2.5, 0, 0.0, 0.0, 0.5, 0.2, 1.0]))
        
        #
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = UR10()
        self.agent.max_velocity = 1.2
        self.joints = [Joint('UR10_joint1'), Joint('UR10_joint2'), Joint('UR10_joint3'), Joint('UR10_joint4'), Joint('UR10_joint5'), Joint('UR10_joint6')]
        self.joints_limits = [[j.get_joint_interval()[1][0],j.get_joint_interval()[1][0]+j.get_joint_interval()[1][1]] 
                              for j in self.joints]
        print(self.joints_limits)
        self.agent_hand = Shape('hand')
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.initial_agent_tip_position = self.agent.get_tip().get_position()
        self.initial_agent_tip_quaternion = self.agent.get_tip().get_quaternion()
        self.target = Dummy('UR10_target')
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.pollo_target = Dummy('pollo_target')
        self.pollo = Shape('pollo')
        self.table_target = Dummy('table_target')
        self.initial_pollo_position = self.pollo.get_position()
        self.initial_pollo_orientation = self.pollo.get_quaternion()
        self.table_target = Dummy('table_target')
        # objects
        self.scene_objects = [Shape('table0'), Shape('Plane'), Shape('Plane0'), Shape('ConcretBlock')]
        #self.agent_tip = self.agent.get_tip()
        self.initial_distance = np.linalg.norm(np.array(self.initial_pollo_position)-np.array(self.initial_agent_tip_position))
        
        # camera 
        self.camera = VisionSensor('kinect_depth')
        self.camera_matrix_extrinsics = vrep.simGetObjectMatrix(self.camera.get_handle(),-1)
        self.np_camera_matrix_extrinsics = np.delete(np.linalg.inv(self.vrepToNP(self.camera_matrix_extrinsics)), 3, 0)
        width = 640.0
        height = 480.0
        angle = math.radians(57.0)
        focalx_px = (width/2.0) / math.tan(angle/2.0)
        focaly_px = (height/2.0) / math.tan(angle/2.0)
        self.np_camera_matrix_intrinsics = np.array([[-focalx_px, 0.0, 320.0],
                                                     [0.0, -focalx_px, 240.0],
                                                     [0.0, 0.0, 1.0]])
                                        
        self.reset()

    def reset(self):
        pos = list(np.random.uniform( [-0.1, -0.1, 0.0],  [0.1, 0.1, 0.1]) + self.initial_pollo_position)
        self.pollo.set_position(pos)
        self.pollo.set_quaternion(self.initial_pollo_orientation)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.initial_epoch_time = time.time()
        while True:         # wait for arm to stop
            self.pr.step()  # Step the physics simulation
            a = self.agent.get_joint_velocities()
            if not np.any(np.where( np.fabs(a) < 0.1, False, True )):
                break
        return self._get_state()

    def step(self, action): 
        print(action.shape, action)
        if action is None:
            self.pr.step()
            return self._get_state(), 0.0, False, {}
        
        if np.any(np.isnan(action)):
            print(action)
            return self._get_state(), -1.0, False, {}

        # create a path with tip and pollo at its endpoints and 5 adjustable points in the middle
        np_pollo_target = np.array(self.pollo_target.get_position())
        np_robot_tip_position = np.array(self.agent.get_tip().get_position())
        np_robot_tip_orientation = np.array(self.agent.get_tip().get_orientation())
        c_path = CartesianPath.create()     
        c_path.insert_control_points(action[4])         # point after pollo to secure the grasp
        c_path.insert_control_points([list(np_pollo_target) + list(np_robot_tip_orientation)])       # pollo
        c_path.insert_control_points(action[0:3])   
        c_path.insert_control_points([list(np_robot_tip_position) + list(np_robot_tip_orientation)]) # robot hand
        try:
            self.path = self.agent.get_path_from_cartesian_path(c_path)
        except IKError as e:
            print('Agent::grasp    Could not find joint values')  
            return self._get_state(), -1, True, {}   
    
        # go through path
        reloj = time.time()
        while self.path.step():
            self.pr.step()                                   # Step the physics simulation
            if (time.time()-reloj) > 4:              
                return self._get_state(), -10.0, True, {}    # Too much time
            if any((self.agent_hand.check_collision(obj) for obj in self.scene_objects)):   # colisión con la mesa
                return self._get_state(), -10.0, True, {}   
        
        # path ok, compute reward
        np_pollo_target = np.array(self.pollo_target.get_position())
        np_robot_tip_position = np.array(self.agent.get_tip().get_position())
        dist = np.linalg.norm(np_pollo_target-np_robot_tip_position)
        altura = np.clip((np_pollo_target[2] - self.initial_pollo_position[2]), 0, 0.5)
        
        if altura > 0.3:                                                     # pollo en mano
            return self._get_state(), altura, True, {}
        elif dist > self.initial_distance:                                   # mano se aleja
            return self._get_state(), -10.0, True, {}
        if time.time() - self.initial_epoch_time > 5:                        # too much time
            return self._get_state(), -10.0, True, {}
        
        # Reward 
        pollo_height = np.exp(-altura*10)  # para 1m pollo_height = 0.001, para 0m pollo_height = 1
        reward = - pollo_height - dist
        return self._get_state(), reward, False, {}

    def close(self):
        self.pr.stop()
        self.pr.shutdown()
    
    def render(self):
        print("RENDER")
        np_pollo_en_camara = self.getPolloEnCamara()
         
        # capture depth image
        depth = self.camera.capture_rgb()
        circ = plt.Circle((int(np_pollo_en_camara[0]), int(np_pollo_en_camara[1])),10)
        plt.clf()
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circ)
        ax.imshow(depth, cmap = 'hot')
        plt.pause(0.000001)

    # Aux

    # transform env.pollo_target.get_position() to camera coordinates and project pollo_en_camera a image coordinates        
    def getPolloEnCamara(self):
        np_pollo_target = np.array(self.pollo_target.get_position())
        np_pollo_target_cam = self.np_camera_matrix_extrinsics.dot(np.append([np_pollo_target],1.0))
        np_pollo_en_camara = self.np_camera_matrix_intrinsics.dot(np_pollo_target_cam)
        np_pollo_en_camara = np_pollo_en_camara / np_pollo_en_camara[2]
        np_pollo_en_camara = np.delete(np_pollo_en_camara,2)
        return np_pollo_en_camara

    def getPolloEnCamaraEx(self):
        np_pollo_target = np.array(self.pollo_target.get_position())
        np_pollo_en_camara = self.np_camera_matrix_extrinsics.dot(np.append([np_pollo_target],1.0))
        return np_pollo_en_camara


    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # return (self.agent.get_joint_positions() + 
        #         self.agent.get_joint_velocities() +
        #         self.pollo_target.get_position())
        p = self.getPolloEnCamaraEx()
        j = self.agent.get_joint_positions()
        r = np.array([j[0],j[1],j[2],j[3],j[4],j[5],p[0],p[1],p[2]])
        return r
    
    def vrepToNP(self, c):
        return np.array([[c[0],c[4],c[8],c[3]],
                         [c[1],c[5],c[9],c[7]],
                         [c[2],c[6],c[10],c[11]],
                         [0,   0,   0,    1]])
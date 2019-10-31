from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.cartesian_path import CartesianPath
import numpy as np
import math, time
from inputs import devices, get_gamepad
import threading
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from pyrep.backend import vrep
import os

SCENE_FILE = '/home/pbustos/software/vrep-simulator/pollos/pollos.ttt'
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]


class Environment(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = UR10()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Dummy('UR10_target')
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.pollo_target = Dummy('pollo_target')
        self.pollo = Shape('pollo')
        self.table_target = Dummy('table_target')
        self.initial_pollo_position = self.pollo.get_position()
        self.initial_pollo_orientation = self.pollo.get_quaternion()
        self.table_target = Dummy('table_target')
        self.initial_agent_tip_position = self.agent.get_tip().get_position()
        self.initial_agent_tip_quaternion = self.agent.get_tip().get_quaternion()
        self.agent_tip = self.agent.get_tip()
        #
        self.camera = VisionSensor('kinect_depth')
       
    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return (self.agent.get_joint_positions() + self.pollo_target.get_position())

    def reset(self):
        pos = list(np.random.uniform( [-0.1, -0.1, 0.0],  [0.1, 0.1, 0.1]) + self.initial_pollo_position)
        self.pollo.set_position(pos)
        self.pollo.set_quaternion(self.initial_pollo_orientation)
        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()

    def step(self, action):
        np_tip = np.array(self.agent_tip.get_position())
        if action is None:
            self.pr.step()
            return 0.0, self._get_state()

        self.agent.set_joint_target_positions(action)
        self.pr.step()  # Step the physics simulation
        
        # Reward is negative distance to target
        #reward = -np.linalg.norm(np_pollo_target-np_tip)
        return 0.0, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

class Agent(object):
    state = "JOYSTICK"

    # simple state machine
    def act(self, env, joy):
        if self.state == "JOYSTICK":
            return(self.joystick(env, joy))
        elif self.state == "INIT_GRASP":
            return(self.init_grasp(env))
        elif self.state == "GRASP":
            return(self.grasp(env))
        elif self.state == "UNLOAD":
            return(self.unload(env))
        elif self.state == "WAIT":
            return(self.wait())
        elif self.state == "RESET_EPISODE":
            return(self.resetEpisode(joy))
            
    
    def joystick(self, env, joy):
        dist = np.linalg.norm(np_robot_tip_position - np_pollo_target)
        angles = env.agent.get_joint_positions()
        if dist < 0.5:
            print("changing to INIT_GRASP")
            self.state = "INIT_GRASP"
            return angles
    
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
        #print(angles)
        print(dist)
        return angles

    def init_grasp(self, env):
        np_pollo_target = np.array(env.pollo_target.get_position())
        np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())
        dist = np.linalg.norm(np_robot_tip_position - np_pollo_target)
        c_path = CartesianPath.create()
        
        # LIFO: goto table
        c_path.insert_control_points([env.table_target.get_position() + list(np_robot_tip_orientation)])
        
        at100 = np.add(np_pollo_target, np.array([0.0,0.0,0.3]))
        c_path.insert_control_points([list(at100) + list(np_robot_tip_orientation)])
        # c_path.insert_control_points([env.table_target.get_position() + list(np_robot_tip_orientation)])
        # np_robot_tip_orientation[0] += 1
        # c_path.insert_control_points([list(np_pollo_target) + list(np_robot_tip_orientation)])
        # np_robot_tip_orientation[0] -= 1
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
        np_robot_tip_orientation = np.array(env.agent.get_tip().get_quaternion())
        dist = np.linalg.norm(np_robot_tip_position - np_pollo_target)
        if self.path.step():
            self.state = "UNLOAD"
        return None

    def unload(self, env):
        angles = env.agent.get_joint_positions()
        angles[5] = math.radians(120)
        self.initReloj(3)
        self.state = "WAIT"
        return angles

    def wait(self):
        if (time.time()-self.reloj) > self.DELAY:
            self.state = "RESET_EPISODE"
        return None

    def resetEpisode(self, joy):
        joy.next_ep = True
        self.state = "JOYSTICK"
        return None

    ########################3
        
    def initReloj(self, delay):
        self.reloj = time.time()
        self.DELAY = delay

    def learn(self, replay_buffer):
        del replay_buffer
        pass

class JoyReader(threading.Thread):
    MAX_ZADV = 20 #mm/sg
    MAX_XADV = 20 #mm/sg
    MAX_YADV = 20 #mm/sg
    end = False
    unloading = False
    next_ep = False
    incs = [0.0, 0.0, 0.0]
    def __init__(self):
	    super(JoyReader, self).__init__(daemon= True)

    def run(self):
        while(True):
            event = get_gamepad()[0]
            #print(event.code,  event.state, event.ev_type)
            if event.code == "ABS_Y":
                self.incs[2] = -(-event.state*self.MAX_ZADV*2/256 + self.MAX_ZADV) / 1000.      # up-down
            elif event.code == "ABS_X":
                self.incs[0] = -(-event.state*self.MAX_XADV*2/256 + self.MAX_XADV) / 1000.      # left-right
            elif event.code == "ABS_RZ":
                self.incs[1] = (-event.state*self.MAX_YADV*2/256 + self.MAX_YADV) / 1000.      # foward-backward
            elif event.code == "BTN_THUMB" and event.state == 1:
                self.next_ep = True
            elif event.code == "BTN_BASE5":
                self.end = True
            elif event.code == "BTN_TRIGGER" and event.state == 1:
                self.unloading = True
            elif event.code == "BTN_TRIGGER" and event.state == 0:
                self.unloading = False   
            
env = Environment()
joy = JoyReader()
agent = Agent()
joy.start()
replay_buffer = []

# read trajectory
# traj = []
# with open('tray.txt', 'r') as f:
#     for line in f:
#         for c in line[1:-3].split(','):
#             traj.append(float(c))
# arm_path = ArmConfigurationPath(env.agent, traj)

plt.figure(1)
fig, ax = plt.subplots(1)

def vrepToNP(c):
        return np.array([[c[0],c[4],c[8],c[3]],
                         [c[1],c[5],c[9],c[7]],
                         [c[2],c[6],c[10],c[11]],
                         [0,   0,   0,    1]])

camera_matrix_extrinsics = vrep.simGetObjectMatrix(env.camera.get_handle(),-1)
np_camera_matrix_extrinsics = np.delete(np.linalg.inv(vrepToNP(camera_matrix_extrinsics)), 3, 0)
width = 640.0
height = 480.0
angle = math.radians(57.0)
focalx_px = (width/2.0) / math.tan(angle/2.0)
focaly_px = (height/2.0) / math.tan(angle/2.0)
np_camera_matrix_intrinsics = np.array([[-focalx_px, 0.0, 320.0],
                                        [0.0, -focalx_px, 240.0],
                                        [0.0, 0.0, 1.0]])
num_ep = 0
c_path = None
grasped = False

while not joy.end:
    failure = False
    joy.next_ep = False
    state = env.reset()
    num_frame = 0
    num_ep += 1
    start = time.time()
    while not joy.next_ep and not joy.end:
        # transform env.pollo_target.get_position() to camera coordinates and project pollo_en_camera a image coordinates
        np_pollo_target = np.array(env.pollo_target.get_position())
        np_pollo_target_cam = np_camera_matrix_extrinsics.dot(np.append([np_pollo_target],1.0))
        np_pollo_en_camara = np_camera_matrix_intrinsics.dot(np_pollo_target_cam)
        np_pollo_en_camara = np_pollo_en_camara / np_pollo_en_camara[2]
        np_pollo_en_camara = np.delete(np_pollo_en_camara,2)
         
        # capture depth image
        depth = env.camera.capture_rgb()
        circ = plt.Circle((int(np_pollo_en_camara[0]), int(np_pollo_en_camara[1])),10)
        plt.clf()
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circ)
        ax.imshow(depth, cmap = 'hot')
        plt.pause(0.000001)

        # save frame
        # end = time.time()
        # if (end-start) > 0.3:
        #     filename = 'pollos-ep-{}-frame-{}'.format(num_ep, num_frame)
        #     print(filename)
        #     num_frame += 1
        #     temp = np.array((depth, (int(np_pollo_en_camara[0]), int(np_pollo_en_camara[1]))))
        #     np.save(filename, temp)
        #     #classes_file.write(filename + " " + f'{int(np_pollo_en_camara[0])}' + " " + f'{int(np_pollo_en_camara[1])}' + os.linesep)
        #     start = time.time()

        # compute trayectory
        np_robot_tip_position = np.array(env.agent.get_tip().get_position())
        np_robot_tip_orientation = np.array(env.agent.get_tip().get_orientation())

        dist = np.linalg.norm(np_robot_tip_position-np_pollo_target)
        # landa = 0.0
        # if c_path is not None:
        #     c_path.remove()
        # c_path = CartesianPath.create()
        # for p in range(int(dist/0.1)):
        #     r = (1.0 - landa) * np_robot_tip_position + (landa * np_pollo_target)
        #     c_path.insert_control_points([list(r) + list(np_robot_tip_orientation)])
        #     landa += 0.1
        
       
        # simulate
        action = agent.act(env, joy)
        reward, next_state = env.step(action)
        replay_buffer.append(action)
        state = next_state
        # reset episode
        # if np.linalg.norm(np_pollo_target) < 0.4:
        #   joy.next_ep = True
       

    print("Resetting environment")
    joy.next_ep = False
    # with open('tray.txt', 'w') as f:
    #     for coor in replay_buffer:
    #         f.write("%s \n" % coor)
    
env.shutdown()


###################################################################









# get_configs_for_tip_pose(position:List[float], euler:List[float]=None, 
#                          quaternion:List[float]=None, 
#                          ignore_collisions=False, trials=300, max_configs=60)

# Reset the arm at the start of each 'episode'
#ur1r
#.set_joint_positions(starting_joint_positions)

# Get a path to the target (rotate so z points down)


# try:
#     print("planning to ", np_rest.tolist())
#     path = ur1r
#.get_linear_path(position=np_rest.tolist())
#     print("done")
    
#     # Step the simulation and advance the agent along the path
#     done = False
#     while not done:
#         done = path.step()
#         pr.step()

#     print('Reached target %d!' % i)
#     time.sleep(1)
# except ConfigurationPathError as e:
#     print('Could not find path')


# done = False
# while not done:   
#     pr.step()
#     np_pollo_target = np.array(pollo_target.get_position())
#     np_tip = np.array(tip.get_position())
#     landa = 0.1
#     np_rest = (1-landa)*np_tip + (landa * np_pollo_target)
#     dist = np.linalg.norm(np_pollo_target-np_tip)
#     #ball.set_position(list(np_rest))
#     print(dist)
#     if dist<0.20:
#         try: 
#             final_pos = np_pollo_target + np.array([0,0.1,0.4])
#             #angles = ur1r
#.solve_ik(position=table_target.get_position(), quaternion=quat)
#             path = ur1r
#.get_path(position=table_target.get_position(),
#                 quaternion=table_target.get_quaternion())
#             #ur1r
#.set_joint_target_positions(angles)
#             done = path.step()
#             #path = ur1r
#.get_configs_for_tip_pose(position=np_rest.tolist(), ignore_collisions=True, trials=300, max_configs=60)
#             done = True
#         except ConfigurationPathError as e:
#             print('Could not find path')    @


# try:
#         # unload_quat = (Quaternion(w=initial_tip_quat[3], x=initial_tip_quat[0], y=initial_tip_quat[1], z=initial_tip_quat[2])
#         #                 * Quaternion(axis=[1.0,0.0,0.0], degrees=60)).unit
#         #unload_quat = (Quaternion(initial_tip_quat)* Quaternion(axis=[0.0,1.0,0.0], degrees=60)).unit
#         #aux = unload_quat[0]
#         #q = unload_quat.elements
#         #q[0]=q[3]
#         #q[3]=aux
#         #print(ur1r
#.get_joint_target_positions())
        
#         angles = ur1r
#.solve_ik(position=tip.get_position(), quaternion=list(unload_quat.elements))
#         ur1r
#.set_joint_target_positions(angles)
#         print(angles)
#     except IKError as e:
#         print('Could not find joint values')
#         continue


 # action = agent.act(state)
        # reward, next_state = env.step(action)
        # replay_buffer.append((state, action, reward, next_state))
        # state = next_state
        # agent.learn(replay_buffer)

        # if joy.unloading:
        #     wrist_angle = math.radians(120)
        # else:
        #     wrist_angle = env.initial_joint_positions[5]
        
        # local_target = np.array(env.agent_tip.get_position()) + np.array(joy.incs)

        #np_pollo_target = np.array(pollo_target.get_position())
        #np_tip = np.array(tip.get_position())
        #dist = np.linalg.norm(np_pollo_target-np_tip)
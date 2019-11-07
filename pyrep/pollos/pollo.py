import numpy as np
import math, time
from joyreader import JoyReader
from joyenv import EnvPollosJoy
from agent import Agent
import threading
import os

env = EnvPollosJoystick()
joy = JoyReader()
agent = Agent()
joy.start()

while not joy.end:
    env.reset()
    joy.next_ep = False
    while not joy.next_ep and not joy.end:
        env.step(joy)
        
    print("Resetting environment")
    joy.next_ep = False
    

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
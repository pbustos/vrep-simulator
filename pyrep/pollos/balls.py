from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.objects.dummy import Dummy
import numpy as np
import math, time

LOOPS = 20
SCENE_FILE = '/home/pbustos/software/vrep/pollos/ur10-only.ttt'
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = UR10()

# We could have made this target in the scene, but lets create one dynamically
target = Shape.create(type=PrimitiveShape.SPHERE,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)

position_min, position_max = [-0.1, -0.1, 0], [0.1, 0.1, 0]

starting_joint_positions = agent.get_joint_positions()

for i in range(LOOPS):

    # Reset the arm at the start of each 'episode'
    agent.set_joint_positions(starting_joint_positions)
    np_tip = np.array(agent.get_tip().get_position())

    # Get a random position within a cuboid and set the target position
    pos = list(np_tip + np.random.uniform(position_min, position_max))
    target.set_position(pos)
    print(pos)
    # Get a path to the target (rotate so z points down)
    try:
        path = agent.get_path(position=pos, euler=[math.radians(180), 0, 0] )
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        done = path.step()
        pr.step()

    print('Reached target %d!' % i)
    time.sleep(0.5)

pr.stop()
pr.shutdown()
import envs
import numpy as np
import matplotlib.pyplot as plt

# 
env = envs.QuadRotorEnv.get_wrapped()
env.reset()
done = False
while not done:
    obs, r, done, _ = env.step(np.array([0, 0, 18]))
    
# 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(
    env.observations_history[0][0],
    env.observations_history[0][1], 
    env.observations_history[0][2])
plt.show()

print('DONE')

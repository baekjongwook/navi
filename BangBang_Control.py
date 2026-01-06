import numpy as np
import matplotlib.pyplot as plt
import math

length = 10.0     
num_points = 50
amplitude = 2.0    
frequency = 0.5 
road_width = 6.0 

x = np.linspace(0, length, num_points)
y = (amplitude * np.sin(frequency * x) + 0.5 * np.sin(2.3 * frequency * x))

y_upper = y + road_width / 2
y_lower = y - road_width / 2

x_r = x[0]
y_r = y[0]
theta = 0.0 

dt = 0.1        
v = 1.0        
omega_max = 2.0  

eps = 0.0    

def y_center_at(xq):
    return np.interp(xq, x, y)

plt.figure(figsize=(16, 8))
plt.fill_between(x, y_lower, y_upper, color='yellow', alpha=0.5, label='Road')
plt.plot(x, y_upper, 'k', linewidth=2)
plt.plot(x, y_lower, 'k', linewidth=2)
plt.plot(x, y, 'b', linewidth=5, alpha=0.5, label='Center Road')
plt.scatter(x[0], y[0], c='green', s=120, zorder=5, label='Start')
plt.scatter(x[-1], y[-1], c='red', s=120, zorder=5, label='Goal')

traj_x, traj_y = [], []
traj_line, = plt.plot([], [], 'r-', linewidth=3, label='Trajectory')
robot_dot, = plt.plot([], [], 'ko', markersize=12, zorder=10)

plt.axis('equal')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bang Bang Control', fontsize = 20)
plt.legend()
plt.ion()
plt.show()

try:
    for _ in range(800):

        if x_r >= x[-1]:
            break

        y_c = y_center_at(x_r)

        e = y_c - y_r

        if e > eps:
            omega = omega_max
        elif e < -eps:
            omega = -omega_max
        else:
            omega = 0.0

        theta += omega * dt
        x_r += v * math.cos(theta) * dt
        y_r += v * math.sin(theta) * dt

        traj_x.append(x_r)
        traj_y.append(y_r)

        traj_line.set_data(traj_x, traj_y)
        robot_dot.set_data([x_r], [y_r])

        plt.pause(0.01)

except KeyboardInterrupt:
    pass

plt.ioff()
input("Press Enter to exit...")
plt.close() 
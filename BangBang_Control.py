import numpy as np
import matplotlib.pyplot as plt

length = 10.0     
num_points = 50
amplitude = 2.0    
frequency = 0.5 
road_width = 6.0 

x = np.linspace(0, length, num_points)
y = (amplitude * np.sin(frequency * x) + 0.5 * np.sin(2.3 * frequency * x))

y_upper = y + road_width / 2
y_lower = y - road_width / 2

# ===== 로봇 초기 상태 =====
x_r = x[0]
y_r = y[0]
theta = 0.0 

plt.figure(figsize=(16, 8))

plt.fill_between(x, y_lower, y_upper, color='yellow', alpha=0.5, label='Road')
plt.plot(x, y_upper, 'k', linewidth=2)
plt.plot(x, y_lower, 'k', linewidth=2)
plt.plot(x, y, 'b', linewidth=5, alpha=0.5, label='Center Road')

plt.scatter(x[0], y[0], c='green', s=120, zorder=5, label='Start')
plt.scatter(x[-1], y[-1], c='red', s=120, zorder=5, label='Goal')

plt.axis('equal')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bang Bang Control', fontsize = 20)
plt.legend()

plt.ion()
plt.show()

arrow_len = 0.1
robot_arrow = None

traj_x = []
traj_y = []
traj_line, = plt.plot([], [], 'r-', linewidth=5, label='Trajectory')

# ===== bang-bang 파라미터 =====
dt = 0.1          # 시간 간격
v = 1.0               # 전진 속도

omega_max = 1.0       # 최대 각속도(핵심)
eps = 0.0           # deadzone(핵심)

def y_center_at(xq):
    return np.interp(xq, x, y)

for i in range(500):
    if x_r >= x[-1]:
        break

    if robot_arrow is not None:
        robot_arrow.remove()

    y_c = y_center_at(x_r)
    e = y_r - y_c

    if e > eps:
        omega = -omega_max
    elif e < -eps:
        omega = omega_max
    else:
        omega = 0.0
        
    theta += omega * dt
    x_r += v * np.cos(theta) * dt
    y_r += v * np.sin(theta) * dt
    
    traj_x.append(x_r)
    traj_y.append(y_r)
    traj_line.set_data(traj_x, traj_y)

    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)

    robot_arrow = plt.arrow(x_r, y_r, dx, dy, head_width=0.4, zorder=6)

    plt.pause(0.001)

plt.ioff()
plt.show()
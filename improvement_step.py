import numpy as np
from shapely import LineString,STRtree,Polygon
from matplotlib import pyplot as plt


lengths = np.array([0.325, 0.275, 0.225])
angle_limits = np.array([
    [-np.pi/2, np.pi/2],
    [-3*np.pi/4, 3*np.pi/4],
    [-3*np.pi/4, 3*np.pi/4]
])

def forward_kinematics(angles, length=lengths):
    cumulative_angles = np.cumsum(angles)
    x = np.sum(length * np.cos(cumulative_angles))
    y = np.sum(length * np.sin(cumulative_angles))
    return np.array([x, y])

def jacobian(angles, length=lengths):
    dof = len(angles)
    J = np.zeros((2, dof))
    cumulative_angles = np.cumsum(angles)
    sin_angles = np.sin(cumulative_angles)
    cos_angles = np.cos(cumulative_angles)
    for i in range(dof):
        J[0, i] = -np.sum(length[i:] * sin_angles[i:])
        J[1, i] = np.sum(length[i:] * cos_angles[i:])
    return J

def inverse_kinematics(target, initial_angles=None, length=lengths,
                       angle_limits=angle_limits, lr=0.1, max_iters=1000, tol=1e-4):
    if initial_angles is None:
        angles = np.zeros(len(length))
    else:
        angles = initial_angles.copy()

    for _ in range(max_iters):
        pos = forward_kinematics(angles, length)
        error = target - pos
        if np.linalg.norm(error) < tol:
             return angles, forward_kinematics(angles, length)
        J = jacobian(angles, length)
        J_pinv = np.linalg.pinv(J)
        angles += lr * (J_pinv @ error)
        # Enforce angle limits
        angles = np.clip(angles, angle_limits[:,0], angle_limits[:,1])
    return None,None



def forward_kinematics_batch(angles, length=np.array([0.325, 0.275, 0.225])):
    n = angles.shape[0]
    agent_locations = np.zeros((n, 4, 2), dtype=np.float32)
    cumulative_angles = np.cumsum(angles, axis=1)
    cos_angles = np.cos(cumulative_angles)
    sin_angles = np.sin(cumulative_angles)
    dx = length * cos_angles
    dy = length * sin_angles
    agent_locations[:, 1, 0] = dx[:, 0]
    agent_locations[:, 1, 1] = dy[:, 0]
    agent_locations[:, 2, 0] = dx[:, 0] + dx[:, 1]
    agent_locations[:, 2, 1] = dy[:, 0] + dy[:, 1]
    agent_locations[:, 3, 0] = dx[:, 0] + dx[:, 1] + dx[:, 2]
    agent_locations[:, 3, 1] = dy[:, 0] + dy[:, 1] + dy[:, 2]
    return agent_locations


def check_movemnt_possible(start,end,length,obstcales,n=1000):

    obstacle_tree = STRtree([obstacle.shape for obstacle in obstcales])

    fks=forward_kinematics_batch(np.linspace(start,end,n),length)
    robot_line=np.array([LineString(i) for i in fks])
    query,_ = obstacle_tree.query(robot_line,"intersects")
    if len(query)>0:
        return False
    return True



def check_direct_movemnt_possible(target,initial_angles,robot_link_lengths,obstcales,n=1000):
    target=np.array(target)+1e-6
    angles_sol, final_pos = inverse_kinematics(target, initial_angles)
    if angles_sol is None:
        print("failed")
        return False,None
    if check_movemnt_possible(initial_angles,angles_sol,robot_link_lengths,obstcales,n):
        return True,angles_sol
    return False,None


if __name__=="__main__":
    
    class Obstacle:
        def __init__(self, x_og, y_og, w, h, buffer=0.025):
            x = x_og - buffer/2
            y = y_og - buffer/2
            self.shape = Polygon([
                (x, y),
                (x, y + h + buffer),
                (x + w + buffer, y + h + buffer),
                (x + w + buffer, y)
            ])

            self.og_shape = Polygon([
                (x_og, y_og),
                (x_og, y_og + h),
                (x_og + w, y_og + h),
                (x_og + w, y_og)
            ])

    obstacles = [
            Obstacle(0.35, 0.25, 0.2, 0.1),
            Obstacle(0.35, -0.35, 0.2, 0.1),
        ]


    initial_angles=[0,0,0]
    target_=[0.7,0.0]
    success,final_angles=check_direct_movemnt_possible(target_,initial_angles,lengths,obstacles,n=1000)
    if success:
        print("sucess")
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot obstacles
        for obstacle in obstacles:
            if obstacle.shape.geom_type == 'Polygon':
                x, y = obstacle.shape.exterior.xy
                ax.fill(x, y, alpha=0.25, fc='green', ec='black', label='Obstacle')
        dis_path=forward_kinematics_batch(np.linspace(initial_angles,final_angles,10),lengths)
        for i in dis_path:
            ax.plot(i[:,0],i[:,1],'r-',linewidth=2)

        plt.show()

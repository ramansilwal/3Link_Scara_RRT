from improvement_step import inverse_kinematics
from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree
import numpy as np
from utlis import forward_kinematics_batch
import os
from shapely import STRtree, LineString



angle_limits = np.array([
    [-np.pi/2, np.pi/2],
    [-3*np.pi/4, 3*np.pi/4],
    [-3*np.pi/4, 3*np.pi/4]
])


angle_path="resoucre/angles.pkl"
if not os.path.exists(angle_path):
    random_angles=np.array([[np.random.uniform (*angle_limits[i]) for i in range(0,3)] for _ in tqdm(range(1_000_000))])
    random_position=forward_kinematics_batch(random_angles)

    with open(angle_path,"wb") as file:
        pickle.dump(random_position,file)
else:
    print("loading pickle")
    with open(angle_path,"rb") as file:
        random_position=pickle.load(file)
        


from improvement_step import inverse_kinematics
from tqdm import tqdm
import random
class RRTAngleBatch:
    def __init__(self, start_pos, obstacles=[], tolerance=0.1, step_size=0.1, batch_size=10,discriminant=0.002):
        self.limits = np.array([
                [-np.pi/2, np.pi/2],
                [-3*np.pi/4, 3*np.pi/4],
                [-3*np.pi/4, 3*np.pi/4]
                        ])
        
        self.length = np.array([0.325, 0.275, 0.225])
        self.workspace_size = sum(self.length)
        self.start_node = np.array(start_pos,dtype=np.float32)
        self.tolerance = tolerance
        self.step_size = step_size
        self.batch_size = batch_size
        self.discriminant = discriminant
        self.tree = {tuple(self.start_node): None}
        self.tree_nodes = [self.start_node]
        self.goal_node = None
        self.obstacles = obstacles
        
        self.kd_tree = KDTree([self.start_node])
        self.load_kinematics_tree()
        self.obstacle_tree = STRtree([obstacle for obstacle in obstacles])
    
    def load_kinematics_tree(self):
        angle_path="resoucre/angles_pos.pkl"
        if not os.path.exists(angle_path):
            print("buliding kinematics tree")
            random_angles=np.array([[np.random.uniform (*angle_limits[i]) for i in range(0,3)] for _ in tqdm(range(1_000_000))])
            random_position=forward_kinematics_batch(random_angles)
            angle_dict={
                "angles": random_angles,
                "positions": random_position,
            }

            with open(angle_path,"wb") as file:
                pickle.dump(angle_dict,file)
        else:
            print("loading pickle")
            with open(angle_path,"rb") as file:
                angle_dict=pickle.load(file)
            random_angles=angle_dict["angles"]
            random_position=angle_dict["positions"]
        self.random_angles=random_angles.copy()
        self.random_position=random_position.copy()
        self.kd_tree_kineamtics=KDTree(random_position[:,-1])
        
                
    def forward_kinematics(self, angles):
        """Single configuration FK for compatibility"""
        result = forward_kinematics_batch(angles.reshape(1, -1), self.length)
        return result[0].T  # Shape: (2, 4)
    
    def sample_positions_batch(self, n, goal_ws=None, goal_bias=0.2):
        """Sample n configurations at once"""
        
        samples=[]
        for i in range(0,n):
            if random.uniform(0,1)>goal_bias:
                samples.append([np.random.uniform(*i) for i in self.limits])
            else:
                samples.append(random.choice(self.potential_angles))
        return np.array(samples)
    
    def nearest_nodes_batch(self, samples):
        """Find nearest node for each sample"""
        dist, idx = self.kd_tree.query(samples, k=1)
        nearest_nodes = self.kd_tree.data[idx]
        return nearest_nodes

    def steer_batch(self, from_nodes, to_nodes):
        """
        Steer from from_nodes toward to_nodes with dynamic step size.
        Step size is proportional to the distance to the target, clipped between min_step and max_step.
        """
        directions = to_nodes - from_nodes
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        
        
        # Compute new nodes
        new_nodes = from_nodes + (directions / (distances + 1e-10)) * self.step_size

        return new_nodes

    def distance_in_workspace_batch(self, angles_batch, goal_ws):
        """Compute workspace distance for batch of configurations"""
        configs = forward_kinematics_batch(angles_batch, self.length)
        end_effectors = configs[:, -1, :]  # Shape: (n, 2)
        distances = np.linalg.norm(end_effectors - goal_ws, axis=1)
        return distances


    def is_in_obstacle_batch(self, angles_batch):
        """Check collision for batch of configurations without loop"""
        
        # Compute all LineStrings at once
        configs = forward_kinematics_batch(angles_batch, self.length)
        lines_batch = np.array([LineString(cfg).buffer(0.05) for cfg in configs])
        
        # Query the obstacle tree
        colliding_indices, _ = self.obstacle_tree.query(lines_batch, predicate="intersects")
        
        # Build boolean collision mask
        collisions = np.zeros(len(lines_batch), dtype=bool)
        collisions[np.unique(colliding_indices)] = True  # mark collisions
        
        return collisions
    
    def build_tree(self, goal_ws, max_iterations=1000, goal_bias=0.2):
        """Build RRT tree toward goal in workspace"""
        goal_ws = np.array(goal_ws, dtype=np.float32)
        mask=self.kd_tree_kineamtics.query_ball_point(goal_ws,self.discriminant)
        self.potential_angles=self.random_angles[mask]
        if len(self.potential_angles)==0:
            print("solution not found")
            return None
        if self.is_in_obstacle_batch(np.array([self.start_node]))[0]:
            print("strat config in collision")
            return None

        for iteration in tqdm(range(max_iterations)):
            # Sample batch of configurations
            samples = self.sample_positions_batch(self.batch_size, goal_ws, goal_bias)
            
            # Find nearest nodes for each sample
            nearest_nodes = self.nearest_nodes_batch(samples)
            
            # Steer toward samples
            new_nodes = self.steer_batch(nearest_nodes, samples)
            
            # Check for duplicates in tree
            valid_mask = np.ones(self.batch_size, dtype=bool)
            for i in range(self.batch_size):
                new_node_tuple = tuple(new_nodes[i])
                if new_node_tuple in self.tree:
                    valid_mask[i] = False
            
            # Filter valid nodes
            if not np.any(valid_mask):
                continue
            
            new_nodes_valid = new_nodes[valid_mask]
            nearest_nodes_valid = nearest_nodes[valid_mask]
            
            # Check collisions
            collisions = self.is_in_obstacle_batch(new_nodes_valid)
            collision_free_mask = ~collisions
            
            if not np.any(collision_free_mask):
                continue
            
            # Add collision-free nodes to tree
            new_nodes_final = new_nodes_valid[collision_free_mask]
            nearest_nodes_final = nearest_nodes_valid[collision_free_mask]
            valid_nodes_final=[]
            valid_nearest_nodes=[]
            for i,j in zip(nearest_nodes_final,new_nodes_final):
                # if simulate_movement(i,j,self.length,self.obstacle_tree,n=10):
                valid_nearest_nodes.append(i)
                valid_nodes_final.append(j)
            for i,j in zip(valid_nearest_nodes,valid_nodes_final):
                new_node_tuple = tuple(j)
                nearest_tuple = tuple(i)
                
                self.tree[new_node_tuple] = nearest_tuple
                self.tree_nodes.append(j)
            
            # Rebuild KD-tree with new nodes
            if iteration%10==0:
                self.kd_tree = KDTree(self.tree_nodes)
            
            distances = np.linalg.norm(new_nodes_final[:, None, :] - self.potential_angles[None, :, :],axis=2)
            
            min_distances = distances.min(axis=1)

            # New nodes close enough to goal configuration
            goal_reached_indices = np.where(min_distances < self.step_size)[0]



            if len(goal_reached_indices) > 0:
                # Pick the closest one
                best_idx = goal_reached_indices[np.argmin(min_distances[goal_reached_indices])]

                self.goal_node = tuple(new_nodes_final[best_idx])

                path = self.reconstruct_path(goal_ws)
                return path

       
        return None

    def reconstruct_path(self,goal_ws=None):
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node)
            node = self.tree[node]
        revered_path=path[::-1]
        if goal_ws is not None:
            ik_angle,pos=inverse_kinematics(goal_ws,np.array(revered_path[-1]))
            if ik_angle is not None:
                revered_path.append(ik_angle)
        

        return revered_path
    
    def render(self, path=None, target=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot obstacles
        for obstacle in self.obstacles:
            shape = getattr(obstacle, "shape", obstacle)
            if shape.geom_type == "Polygon":
                x, y = shape.exterior.xy
                ax.fill(x, y, alpha=0.25, fc="green", ec="black", label="Obstacle")
            elif shape.geom_type == "LineString":
                x, y = shape.xy
                ax.plot(x, y, color="black", linewidth=1.0)

            og_shape = getattr(obstacle, "og_shape", None)
            if og_shape is not None and og_shape.geom_type == "Polygon":
                x, y = og_shape.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Obstacle")

        # Plot tree
        for node, parent_node in self.tree.items():
            if parent_node is None:
                continue
            parent = self.forward_kinematics(np.array(parent_node))
            child = self.forward_kinematics(np.array(node))
            ax.plot(
                [parent[0, -1], child[0, -1]],
                [parent[1, -1], child[1, -1]],
                color="lightblue",
                linewidth=0.3,
                alpha=0.5,
            )

        # Plot path
        if path is not None:
            path=np.array(path)
            count=0
            for i in forward_kinematics_batch(path, self.length):
                ax.plot(i[:,0],i[:,1])
                count+=1
                ax.text(i[-1][0],i[-1][1],f"{count}")

        # Plot target
        if target is not None:
            ax.scatter(target[0],target[1],marker="x")
        
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'RRT Path Planning (Tree size: {len(self.tree)} nodes)')
        
        return fig


import numpy as np
from shapely.geometry import LineString

def smooth_path(path, rrt, robot_radius=0.05, num_checks=100):
    """
    Smooth a joint-space RRT path by removing unnecessary waypoints.

    path: list/array of joint angle configurations
          Example: [q0, q1, q2, ...]
    rrt: object containing:
         - rrt.obstacle_tree
         - rrt.length
    """

    if path is None or len(path) == 0:
        return []

    if len(path) <= 2:
        return path

    obstacles_tree = rrt.obstacle_tree
    path = [np.asarray(p, dtype=float) for p in path]

    new_path = [path[0]]
    i = 0
    n = len(path)

    def collision_free(q_start, q_end):
        # Interpolate in joint space
        configs = np.linspace(q_start, q_end, num_checks)

        # FK gives robot joint locations for each interpolated config
        robot_points = forward_kinematics_batch(configs, rrt.length)

        for pts in robot_points:
            robot_shape = LineString(pts).buffer(robot_radius)

            hits = obstacles_tree.query(robot_shape, predicate="intersects")

            if len(hits) > 0:
                return False

        return True

    while i < n - 1:
        # Try to jump as far forward as possible
        best_j = i + 1

        for j in range(i + 2, n):
            if collision_free(path[i], path[j]):
                best_j = j
            else:
                break

        new_path.append(path[best_j])
        i = best_j

    return new_path

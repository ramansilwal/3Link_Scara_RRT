import numpy as np


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


import numpy as np

def distance_to_line_segment(point, a, b):
    point = np.array(point, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    ab = b - a
    ap = point - a

    # projection factor of point onto line AB
    t = np.dot(ap, ab) / np.dot(ab, ab)

    # clamp t so closest point stays on the segment
    t = np.clip(t, 0, 1)

    closest_point = a + t * ab

    distance = np.linalg.norm(point - closest_point)

    return distance, closest_point

def process_laser(laser,robot,distance_filter=0.4,upper_range=1.5,min_intensity=5):
        
    jp={x:y for x,y in zip(robot["name"],robot["position"])}
    position=forward_kinematics_batch(np.array([[jp["Angle1"],jp["Angle2"],jp["Angle3"]]]))[0]


    count=0
    angle=laser["angle_min"]
    pos=[]

    while angle<laser["angle_max"]+laser["angle_increment"]:
        x=np.cos(angle)*laser["ranges"][count]
        y=np.sin(angle)*laser["ranges"][count]
        if laser["intensities"][count]>min_intensity and np.linalg.norm([x,y])<upper_range:
            pos.append([x,y])
        # print(count,angle,laser["ranges"][count],laser["intensities"][count])

        count+=1
        angle+=laser["angle_increment"]
    pos=np.array(pos)
    pos_true=np.array([[pos[i,0]+position[1][0],-pos[i,1]+position[1][1]] for i in range(0,len(pos))])
    pos_filtered=pos_true[np.linalg.norm(pos_true,axis=1)>0.2]


    p=[]
    for point in pos_filtered:
        distance,closet_point=(distance_to_line_segment(point, position[-2], position[-1]))
        p.append(distance)


    points_selected=[]
    for point in pos_filtered:
        distance,closet_point=(distance_to_line_segment(point, position[-2], position[-1]))
        if distance>distance_filter:
            points_selected.append(point)
    
    return np.array(points_selected)
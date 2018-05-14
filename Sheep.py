import numpy as np
import matplotlib.pyplot as plt
from importJSON import Map
from Animal import Animal

# Global variables defining the sheep behavior
SHEEP_R = 0.7
# The space the sheep wants between them
SPACE_R = 3* SHEEP_R
RANGE_R = 3* SHEEP_R

BUMP_h = 0.2  # Value from the paper. Used in the bump function
A = 5
B = 5
C = abs(A-B)/(np.sqrt(4*A*B))
EPS = 0.1


# ---------- The functions from the paper by Harman. Used for computing the agent acceleration

# Separation

O = 1

S = 0.8

K = 0.5
M = 0.4



def separation(agent, neighbors):
    """
    A function to calculate the separation steer for an agent
    :param agent: the agent
    :param neighbors: other agents in the visible set
    :return:
    """
    s = np.zeros(2)  # separation_steer
    for neighbor in neighbors:
        #if np.linalg.norm(neighbor.pos - agent.pos) < SPACE_R:
        s -= (neighbor.pos - agent.pos) / np.linalg.norm(neighbor.pos - agent.pos)
        #s -= (agent.pos - neighbor.pos)
    return s

def cohesion(agent, neighbors):
    """
    Calculates the cohesion displacement vector fpr the agent
    :param agent: the agent
    :param neigbors: other agents in the visible set
    :return:
    """
    c = np.zeros(2)  # center of the visible set
    if len(neighbors) > 0:
        for neigbor in neighbors:
            c += neigbor.pos
        c /= len(neighbors)

        k = c - agent.pos  # cohesion displacement vector

        return k
    else:
        return c

def alignment(neighbors):
    """
    Calculates the alignment (velocity matching)
    :param agent: the agent
    :param neighbors:other agents in the visible set
    :return:
    """
    m = np.zeros(2)  # separation_steer

    if len(neighbors) > 0:
        for neighbor in neighbors:
            m += neighbor.vel
        m /= len(neighbors)

    return m

def obstacle_avoidance(agent, obstacle_sheep):
    o = np.zeros(2)
    if obstacle_sheep is not None:
        o = obstacle_sheep.vel/np.linalg.norm(agent.pos - obstacle_sheep.pos) * 2
    return o

def get_velocity(agent, neighbors, obstacle):
    """
    Returns the new velocity based on the neighbors
    :param agent:
    :param neighbors:
    :return:
    """
    s = separation(agent, neighbors)
    k = cohesion(agent, neighbors)
    m = alignment(neighbors)
    o = obstacle_avoidance(agent, obstacle)
    #if len(obstacle) > 0:
    #    o = obstacles[0].vel
    #    #print(o)
    #else:
    #    o = 0

    new_vel = agent.vel + S*s + K*k + M*m + O*o

    if np.linalg.norm(new_vel) > agent.max_vel:
        new_vel /= np.linalg.norm(new_vel)
        new_vel *= agent.max_vel

    return new_vel
    #return agent.vel + O*o


class Sheep(Animal):
    def __init__(self, the_map, pos, vel=np.zeros(2)):
        super().__init__(pos, vel, the_map.sheep_r, the_map.sheep_v_max, the_map.sheep_a_max,
                         the_map.sheep_sight_range, the_map.sheep_sight_ang)

    def get_acceleration(self, neighbors, obstacles, dt):
        # The gradient based term: finding the best position
        # Consensus term: Tries to adapt the velocity to the neighbors
        new_vel = get_velocity(self, neighbors, obstacles)

        acc = (new_vel - self.vel)/dt
        return acc

    def find_new_vel(self, neighbors, obstacles, dogs, dt):
        new_acc = self.get_acceleration(neighbors, obstacles, dt)
        self.set_new_vel(new_acc, dt)


def find_neighbors(sheep_list, the_sheep):
    neighbors = []
    for list_sheep in sheep_list:
        if list_sheep == the_sheep:
            continue
        elif np.linalg.norm(list_sheep.pos - the_sheep.pos) < RANGE_R:
            neighbors.append(list_sheep)
    print(len(neighbors))
    return neighbors

def plot_sheep(sheep):
    plt.clf()
    plt.plot(0,0,"o")
    plt.plot(10,0,"o")
    plt.plot(0,10,"o")
    plt.plot(10,10,"o")
    for a_sheep in sheep:
        # Plots the position
        plt.plot(a_sheep.pos[0], a_sheep.pos[1], "o")

        # Plot velocity
        plt.plot([a_sheep.pos[0], a_sheep.vel[0] + a_sheep.pos[0]], [a_sheep.pos[1], a_sheep.vel[1] + a_sheep.pos[1]])
    plt.pause(0.05)
    #plt.show()

def test():
    the_map = Map("../maps/M1.json")
    sheep1 = Sheep(the_map, np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    sheep2 = Sheep(the_map, np.array([2.0, 2.0]), np.array([-0.0, -0.0]))
    sheep3 = Sheep(the_map, np.array([2.3, 2.3]), np.array([-0.0, -0.0]))
    sheep4 = Sheep(the_map, np.array([1.8, 2.0]), np.array([-0.0, -0.0]))
    sheep5 = Sheep(the_map, np.array([1.0, 2.0]), np.array([1.0, 0]))

    sheep_list = [sheep1, sheep2, sheep3, sheep4, sheep5]


    plot_sheep([sheep1, sheep2, sheep3, sheep4, sheep5])

    for timestep in range(1000):
        for sheep in sheep_list:
            sheep.find_new_vel(find_neighbors(sheep_list, sheep), [], [], [], 0.1)

        for sheep in sheep_list:
            sheep.update(0.1)

        plot_sheep([sheep1, sheep2, sheep3, sheep4, sheep5])

    plt.show()

#test()



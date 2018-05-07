import numpy as np
import matplotlib.pyplot as plt

# Global variables defining the sheep behavior
SHEEP_R = 0.7
# The space the sheep wants between them
SPACE_R = SHEEP_R
RANGE_R = 3 * SPACE_R

BUMP_h = 0.2  # Value from the paper. Used in the bump function
A = 5
B = 5
C = abs(A-B)/(np.sqrt(4*A*B))
EPS = 0.1


# ---------- The functions from the paper by Harman. Used for computing the agent acceleration

# Separation
S = 0.5

K = 0.5
M = 0.5


def separation(agent, neighbors):
    """
    A function to calculate the separation steer for an agent
    :param agent: the agent
    :param neighbors: other agents in the visible set
    :return:
    """
    s = np.zeros(2)  # separation_steer
    for neigbor in neighbors:
        s -= (agent.pos - neigbor.pos)
    return s

def cohesion(agent, neighbors):
    """
    Calculates the cohesion displacement vector fpr the agent
    :param agent: the agent
    :param neigbors: other agents in the visible set
    :return:
    """
    c = np.zeros(2)  # center of the visible set
    for neigbor in neighbors:
        c += neigbor.pos
    c /= len(neighbors)

    k = c - agent.pos  # cohesion displacement vector

    return k


def alignment(neighbors):
    """
    Calculates the alignment (velocity matching)
    :param agent: the agent
    :param neighbors:other agents in the visible set
    :return:
    """
    m = np.zeros(2)  # separation_steer

    if len(neighbors) > 0:
        for neigbor in neighbors:
            m += neigbor.vel
        m /= len(neighbors)

    return m

def get_velocity(agent, neighbors):
    """
    Returns the new velocity based on the neighbors
    :param agent:
    :param neighbors:
    :return:
    """
    s = separation(agent, neighbors)
    k = cohesion(agent, neighbors)
    m = alignment(neighbors)
    return agent.vel + S*s + K*k + M*m


class Sheep:
    def __init__(self, pos, vel=np.zeros(2)):
        self.radius = SHEEP_R
        self.pos = pos
        self.vel = vel
        self.pos_hist = []

    def get_acceleration(self, neighbors):
        # The gradient based term: finding the best position
        # Consensus term: Tries to adapt the velocity to the neighbors
        new_vel = get_velocity(self, neighbors)
        acc = (new_vel - self.vel)/0.1
        return acc

    def update(self, neighbors):
        # Get the acceleration
        # Update the velocity
        # Update the position
        new_acc = self.get_acceleration(neighbors)
        if np.linalg.norm(new_acc) > 1:
            # Scale the acc vector
            new_acc /= np.linalg.norm(new_acc)
        self.vel += new_acc * 0.1
        if np.linalg.norm(self.vel) > 10:
            self.vel /= np.linalg.norm(self.vel)
            self.vel *= 10
        self.pos_hist.append(self.pos)
        self.pos += self.vel * 0.1


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

def test():
    sheep1 = Sheep(np.array([1.0, 1.0]), np.array([0.1, 0.1]))
    sheep2 = Sheep(np.array([2.0, 2.0]), np.array([0.1, -0.1]))
    sheep3 = Sheep(np.array([2.3, 2.3]), np.array([0.1, -0.1]))
    sheep4 = Sheep(np.array([1.8, 2.0]), np.array([0.1, -0.1]))
    sheep5 = Sheep(np.array([1.0, 2.0]), np.array([0.1, 0.0]))

    sheep_list = [sheep1, sheep2, sheep3, sheep4, sheep5]


    plot_sheep([sheep1, sheep2, sheep3, sheep4, sheep5])

    for timestep in range(1000):
        sheep1.update(find_neighbors(sheep_list, sheep1))
        sheep2.update(find_neighbors(sheep_list, sheep2))
        sheep3.update(find_neighbors(sheep_list, sheep3))
        sheep4.update(find_neighbors(sheep_list, sheep4))
        sheep5.update(find_neighbors(sheep_list, sheep5))

        plot_sheep([sheep1, sheep2, sheep3, sheep4, sheep5])

    plt.show()

test()



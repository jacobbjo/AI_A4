import numpy as np
import matplotlib.pyplot as plt
from importJSON import Map
from Animal import Animal
from math import *

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

O = 2
F = 0.5

S = 0.4

K = 0.3
M = 0.2

class Sheep(Animal):
    def __init__(self, the_map, pos, vel=np.zeros(2)):
        super().__init__(pos, vel, the_map.sheep_r, the_map.sheep_v_max, the_map.sheep_a_max,
                         the_map.sheep_sight_range, the_map.sheep_sight_ang)

    def get_acceleration(self, neighboring_squares, obstacles, dogs, dt):
        # The gradient based term: finding the best position
        # Consensus term: Tries to adapt the velocity to the neighbors
        new_vel = self.get_velocity(neighboring_squares, obstacles, dogs)

        acc = (new_vel - self.vel)/dt
        return acc

    def find_new_vel(self, neighboring_squares, obstacles, dogs, dt):
        near_dogs = self.analyze_dogs(dogs)
        new_acc = self.get_acceleration(neighboring_squares, obstacles, near_dogs, dt)
        self.set_new_vel(new_acc, dt)

    def analyze_dogs(self, all_dogs):
        near_dogs = []
        for dog in all_dogs:
            if np.linalg.norm(self.pos - dog.pos) < self.sight_range * 10:
                near_dogs.append(dog)
        return near_dogs


    def separation(self, neighbors):
        """
        A function to calculate the separation steer for an agent
        :param self: the agent
        :param neighbors: other agents in the visible set
        :return:
        """
        s = np.zeros(2)  # separation_steer
        for neighbor in neighbors:
            # if np.linalg.norm(neighbor.pos - agent.pos) < SPACE_R:
            s -= (neighbor.pos - self.pos) / np.linalg.norm(neighbor.pos - self.pos)
            # s -= (agent.pos - neighbor.pos)
        return s

    def cohesion(self, neighbors):
        """
        Calculates the cohesion displacement vector fpr the agent
        :param self: the agent
        :param neigbors: other agents in the visible set
        :return:
        """
        c = np.zeros(2)  # center of the visible set
        if len(neighbors) > 0:
            for neighbor in neighbors:
                c += neighbor.pos
            c /= len(neighbors)

            k = c - self.pos  # cohesion displacement vector

            return k
        else:
            return c

    def alignment(self, neighbors):
        """
        Calculates the alignment (velocity matching)
        :param self: the agent
        :param neighbors:other agents in the visible set
        :return:
        """
        m = np.zeros(2)  # separation_steer

        if len(neighbors) > 0:
            for neighbor in neighbors:
                m += neighbor.vel - self.vel
            m /= len(neighbors)

        return m

    def obstacle_avoidance(self, obstacle_sheep):

        o = np.zeros(2)
        if obstacle_sheep is not None:
            scale = np.linalg.norm(self.pos - obstacle_sheep.pos) * 2
            if scale > 0:
                o += obstacle_sheep.vel / scale
            else:
                o += obstacle_sheep.vel
        return o

        # ---- For more agents in a list
        #for obstacle_agent in obstacle_sheep:
        #    if obstacle_agent is not None:
        #        scale = np.linalg.norm(self.pos - obstacle_agent.pos) * 2
        #        if scale > 0:
        #            o += obstacle_agent.vel / scale
        #        else:
        #            o += obstacle_agent.vel
        #return o


    def avoid_dogs(self, dogs):
        f = np.zeros(2)
        for dog in dogs:
            away_vec = self.pos - dog.pos
            away_vel = (away_vec / np.linalg.norm(away_vec)) * (self.max_vel / np.linalg.norm(away_vec))
            f += away_vel
        if len(dogs) > 0:
            f /= len(dogs)

        return f

    def get_velocity(self, neighboring_squares, obstacles, dogs):
        """
        Returns the new velocity based on the neighbors
        :param self:
        :param neighbors:
        :return:
        """

        obstacle = self.get_obstacle_agents(obstacles)
        neighbors = self.get_neighbors_in_sight(neighboring_squares)

        s = self.separation(neighbors)
        k = self.cohesion(neighbors)
        m = self.alignment(neighbors)
        o = self.obstacle_avoidance(obstacle)
        f = self.avoid_dogs(dogs)

        new_vel = self.vel + S * s + K * k + M * m + O * o + F * f

        if np.linalg.norm(new_vel) > self.max_vel:
            new_vel /= np.linalg.norm(new_vel)
            new_vel *= self.max_vel

        return new_vel


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
            sheep.find_new_vel(find_neighbors(sheep_list, sheep), [], [], 0.1)

        for sheep in sheep_list:
            sheep.update(0.1)

        plot_sheep([sheep1, sheep2, sheep3, sheep4, sheep5])

    plt.show()

#test()



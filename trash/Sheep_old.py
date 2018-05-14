import numpy as np
import matplotlib.pyplot as plt

# Global variables defining the sheep properties
#SHEEP_R = 0
#ANGLE_OF_SIGHT = 3 * (np.pi / 2)
# The space the sheep wants between them
#SPACE_R = SHEEP_R
#RANGE_R = 2 * SPACE_R

BUMP_h = 0  # Value from the paper. Used in the bump function
A = 5
B = 5
C = abs(A-B)/(np.sqrt(4*A*B))
EPS = 0.1


OBSTACLE_R = 5
BUMP_h_beta = 0.9

# Constants to determine the importance
C1_alpha = 1
C2_alpha = 1
C1_beta = 8
C2_beta = 5


# ---------- The functions from the paper. Used for computing the agent acceleration
def bump_function(h, z):
    if 0 <= z < h:
        return 1
    if h <= z <= 1:
        div = ((z-h)/(1-h))
        cos_term = np.cos(np.pi * ((z-h)/(1-h)))
        result = 0.5 * (1 + cos_term)
        return result
    return 0


def sigmoid(z):
    return 0.5 * ((A + B) * gradient_function(1, z + C) + (A - B))


def sigma_norm(z):
    return (1/EPS) * ((np.sqrt(1 + (EPS * (np.linalg.norm(z)**2)))) - 1)


def gradient_function(eps, z):
    if eps == 1:
        return z/(np.sqrt(1 + z ** 2))
    return z / (1 + eps * sigma_norm(z))


def adjacency_value(pos_i, pos_j, sheep):
    return bump_function(BUMP_h, sigma_norm(pos_j - pos_i)/sigma_norm(sheep.sight_range))


def action_function(z, sheep):
    """

    :param z: The value
    :param radius: The vector between the agents, from i to j (pos_i - pos_j)
    :return: The action function value
    """
    sig_norm = sigma_norm(sheep.sight_range)
    sigm = sigmoid(z - sheep.space)
    div = z / sig_norm
    bump = bump_function(BUMP_h, z / sig_norm) * sigm
    return bump

# The beta functions
def action_function_beta(z):
    # Repulsive function
    sigm = gradient_function(1, (z-sigma_norm(OBSTACLE_R))) - 1
    bump = bump_function(BUMP_h_beta, z/sigma_norm(OBSTACLE_R))
    return bump * sigm

def adjacency_value_beta(pos_i, pos_k):
    return bump_function(BUMP_h_beta, (sigma_norm(pos_k - pos_i))/sigma_norm(OBSTACLE_R))


class Sheep:
    def __init__(self, the_map, pos, vel=np.zeros(2)):
        self.radius = the_map.sheep_r

        self.pos = pos
        self.vel = vel
        self.max_vel = the_map.sheep_v_max
        self.max_acc = the_map.sheep_a_max
        if np.linalg.norm(vel) == 0:
            rand_pos = np.random.normal(0.5, 0.5, 2)
            self.dir = rand_pos/np.linalg.norm(rand_pos)
        else:
            self.dir = vel/np.linalg.norm(vel)
        self.pos_hist = []
        self.sight_ang = 3 * (np.pi / 2)
        self.space = self.radius * 3
        self.sight_range = self.radius * 5


        # Test to not update the sheep directly
        self.next_vel = self.vel

    def get_acceleration(self, neighbors, close_neighbors, obstacles, dogs):
        # The gradient based term: finding the best position
        # Consensus term: Tries to adapt the velocity to the neighbors
        # If there are no neighbors we can stop

        # NEED TO ADD CONSTANTS BEFORE GRADIENT AND CONSENSUS, depending on their importance
        # The sheep neighbors
        #if len(neighbors) == len(obstacles) == 0:
        #    return -self.vel
        gradient = np.zeros(2)
        consensus = np.zeros(2)

        for neighbor in neighbors:
            gradient += action_function(sigma_norm(((neighbor.pos - self.pos)/2)), self) * \
                        gradient_function(EPS, (neighbor.pos - self.pos))

            consensus += adjacency_value(self.pos, neighbor.pos, self) * (neighbor.vel - self.vel)

        # The obstacle neighbors
        gradient_obstacle = np.zeros(2)
        consensus_obstacle = np.zeros(2)
        for obstacle in obstacles:
            gradient_obstacle += action_function_beta(sigma_norm(obstacle.pos - self.pos)) * \
                        gradient_function(EPS, obstacle.pos - self.pos)  # / len(neighbors)
            consensus_obstacle += adjacency_value_beta(self.pos, obstacle.pos) * (obstacle.vel - self.vel)

        acceleration = C1_alpha * gradient + C2_alpha * consensus #+ \
                       #C1_beta * gradient_obstacle + C2_beta * consensus_obstacle

        return acceleration

    # Old update function, keep for now in case the other f**cks up.
    #def update(self, neighbors, obstacles):
    #    # Get the acceleration
    #    # Update the velocity
    #    # Update the position
    #    new_acc = self.get_acceleration(neighbors, obstacles)
    #    if np.linalg.norm(new_acc) > 1:
    #        # Scale the acc vector
    #        new_acc /= np.linalg.norm(new_acc)
    #    self.vel += new_acc * 0.1
    #    if np.linalg.norm(self.vel) > 10:
    #        self.vel /= np.linalg.norm(self.vel)
    #        self.vel *= 10
    #    if np.linalg.norm(self.vel) > 0:
    #        self.dir = self.vel / np.linalg.norm(self.vel)
    #    self.pos_hist.append(self.pos)
    #    self.pos += self.vel * 0.1

    def find_new_vel(self, neighbors, close_neighbors,  obstacles, dogs, dt):
        new_acc = self.get_acceleration(neighbors, close_neighbors, obstacles, dogs)
        if np.linalg.norm(new_acc) > self.max_acc:
            # Scale the acc vector
            new_acc /= np.linalg.norm(new_acc)

        self.next_vel += new_acc * dt

        if np.linalg.norm(self.next_vel) > self.max_vel:
            self.next_vel /= np.linalg.norm(self.next_vel)
            self.next_vel *= self.max_vel

    def update(self, dt):
        self.vel = self.next_vel
        if np.linalg.norm(self.vel) > 0:
            self.dir = self.vel / np.linalg.norm(self.vel)
        self.pos_hist.append(self.pos)
        self.pos += self.vel * dt

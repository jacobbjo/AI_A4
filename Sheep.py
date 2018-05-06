import numpy as np

# Global variables defining the sheep behavior
SHEEP_R = 0.7
# The space the sheep wants between them
SPACE_R = 2 * SHEEP_R

BUMP_h = 0.2  # Value from the paper. Used in the bump function
A = 5
B = 5
C = abs(A-B)/(np.sqrt(4*A*B))
EPS = 0.1


# ---------- The functions from the paper. Used for computing the agent acceleration
def bump_function(z):
    if 0 <= z < BUMP_h:
        return 1
    if BUMP_h <= z <= 1:
        return 0.5 * (1 + np.cos(np.pi * ((z-BUMP_h)/(1-BUMP_h))))
    return 0


def sigmoid(z):
    return 0.5 * ((A + B) * gradient_function(1, z + C) + (A - B))


def sigma_norm(z):
    return (1/EPS) * ((np.sqrt(1 + (EPS * (np.linalg.norm(z)**2)))) - 1)


def gradient_function(eps, z):
    if eps == 1:
        return z/(np.sqrt(1 + z ** 2))
    return z / (1 + eps * sigma_norm(z))


def adjacency_value(pos_i, pos_j, radius):
    return bump_function(sigma_norm(pos_j - pos_i)/sigma_norm(radius))


def action_function(z, radius):
    """

    :param z: The value
    :param radius: The vector between the agents, from i to j (pos_i - pos_j)
    :return: The action function value
    """
    return bump_function(z / sigma_norm(radius)) * sigmoid(z - SPACE_R)


class Sheep:
    def __init__(self, pos):
        self.radius = SHEEP_R
        self.pos = pos
        self.vel = np.zeros(2)

    def get_acceleration(self, neighbors):
        # The gradient based term: finding the best position
        # Consensus term: Tries to adapt the velocity to the neighbors
        gradient = 0
        consensus = 0
        for neighbor in neighbors:
            gradient += action_function(sigma_norm(neighbor.pos - self.pos), self.pos - neighbor.pos) * \
                        gradient_function(EPS, neighbor.pos - self.pos)
            consensus += adjacency_value(self.pos, neighbor.pos, self.pos - neighbor.pos) * (neighbor.vel - self.vel)

        acceleration = gradient + consensus

        return acceleration

    def update(self, neighbors):
        # Get the acceleration
        # Update the velocity
        # Update the position
        print("hej")

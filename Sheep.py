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


# ---------- The functions from the paper. Used for computing the agent acceleration
def bump_function(z):
    if 0 <= z < BUMP_h:
        return 1
    if BUMP_h <= z <= 1:
        div = ((z-BUMP_h)/(1-BUMP_h))
        cos_term = np.cos(np.pi * ((z-BUMP_h)/(1-BUMP_h)))
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


def adjacency_value(pos_i, pos_j, radius):
    return bump_function(sigma_norm(pos_j - pos_i)/sigma_norm(RANGE_R))


def action_function(z, radius):
    """

    :param z: The value
    :param radius: The vector between the agents, from i to j (pos_i - pos_j)
    :return: The action function value
    """
    sig_norm = sigma_norm(RANGE_R)
    sigm = sigmoid(z - SPACE_R)
    div = z / sig_norm
    bump = bump_function(z / sig_norm) * sigm
    return bump


class Sheep:
    def __init__(self, pos, vel=np.zeros(2)):
        self.radius = SHEEP_R
        self.pos = pos
        self.vel = vel

    def get_acceleration(self, neighbors):
        # The gradient based term: finding the best position
        # Consensus term: Tries to adapt the velocity to the neighbors
        gradient = np.zeros(2)
        consensus = np.zeros(2)
        for neighbor in neighbors:
            gradient += action_function(sigma_norm(neighbor.pos - self.pos), self.pos - neighbor.pos) * \
                        gradient_function(EPS, neighbor.pos - self.pos)
            consensus += adjacency_value(self.pos, neighbor.pos, self.pos - neighbor.pos) * (neighbor.vel - self.vel)

        acceleration = gradient + consensus
        acc_norm = np.linalg.norm(acceleration)
        return acceleration

    def update(self, neighbors):
        # Get the acceleration
        # Update the velocity
        # Update the position
        print("hej")

def plot_sheep(sheep):
    for a_sheep in sheep:
        # Plots the position
        plt.plot(a_sheep.pos[0], a_sheep.pos[1], "o")

        # Plot velocity
        plt.plot([a_sheep.pos[0], ], [a_sheep.pos[1], ])


    plt.show()


sheep1 = Sheep(np.array([1,1]), np.array([0.7,0.7]))
sheep2 = Sheep(np.array([2,2]), np.array([0.7,-0.7]))
sheep3 = Sheep(np.array([2.3,2.3]), np.array([0.7,-0.7]))
sheep4 = Sheep(np.array([1.8,2]), np.array([0.7,-0.7]))

sheep1.get_acceleration([sheep2, sheep3, sheep4])

plot_sheep([sheep1, sheep2, sheep3, sheep4])




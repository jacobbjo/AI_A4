import numpy as np

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
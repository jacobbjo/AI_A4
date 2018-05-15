import numpy as np
from Animal import Animal


def obstacle_avoidance(agent, obstacle_sheep):
    o = np.zeros(2)
    if obstacle_sheep is not None:
        o = obstacle_sheep.vel/np.linalg.norm(agent.pos - obstacle_sheep.pos) * 2
    return o


class Dog(Animal):

    def __init__(self, the_map, position, velocity = np.zeros(2)):
        super().__init__(position, velocity, the_map.dog_r, the_map.dog_v_max, the_map.dog_a_max,
                         the_map.dog_sight_range, the_map.dog_sight_ang)


    def get_acceleration(self, obstacles, dt):
        obstacle = self.get_obstacle_agents(obstacles)
        o = obstacle_avoidance(self, obstacle)
        new_vel = self.vel + o
        if np.linalg.norm(new_vel) > self.max_vel:
            new_vel /= np.linalg.norm(new_vel)
            new_vel *= self.max_vel

        acc = (new_vel - self.vel)/dt
        return acc

    def find_new_vel(self, obstacles, dt):
        new_acc = self.get_acceleration(obstacles, dt)
        self.set_new_vel(new_acc, dt)

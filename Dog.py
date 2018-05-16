import numpy as np
from Animal import Animal




class Dog(Animal):

    def __init__(self, the_map, position, velocity = np.zeros(2)):
        super().__init__(position, velocity, the_map.dog_r, the_map.dog_v_max, the_map.dog_a_max,
                         the_map.dog_sight_range, the_map.dog_sight_ang)
        self.right_bound = 0
        self.left_bound = 0

        towards_right = True


    def get_acceleration(self, obstacles, dt):
        """Finds the new acceleration"""
        obstacle = self.get_obstacle_agents(obstacles)
        o = self.obstacle_avoidance(obstacle)
        new_vel = self.vel + o
        if np.linalg.norm(new_vel) > self.max_vel:
            new_vel /= np.linalg.norm(new_vel)
            new_vel *= self.max_vel

        acc = (new_vel - self.vel)/dt
        return acc

    def find_new_vel(self, obstacles, dt):
        """Finds the new velocity given the new acceleration"""
        new_acc = self.get_acceleration(obstacles, dt)
        self.set_new_vel(new_acc, dt)

    def obstacle_avoidance(self, obstacle_sheep):
        """Returns the velocity for avoiding an obstacle"""
        o = np.zeros(2)
        if obstacle_sheep is not None:
            o = obstacle_sheep.vel / np.linalg.norm(self.pos - obstacle_sheep.pos) * 2
        return o





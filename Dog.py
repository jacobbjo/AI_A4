import numpy as np
from Animal import Animal

class Dog(Animal):

    def __init__(self, the_map, position, velocity = np.zeros(2)):
        super().__init__(position, velocity, the_map.dog_r, the_map.dog_v_max, the_map.dog_1_max)


    def get_acceleration(self):

    def find_new_vel(self, dt):

        new_acc = self.get_acceleration(neighbors, obstacles, dt)

        self.set_new_vel(new_acc, dt)

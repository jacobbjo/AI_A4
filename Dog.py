import numpy as np
from Animal import Animal
from math import *
import matplotlib.pyplot as plt

O = 0
H = 1


class Dog(Animal):

    def __init__(self, the_map, position, velocity = np.zeros(2)):
        super().__init__(position, velocity, the_map.dog_r, the_map.dog_v_max, the_map.dog_a_max,
                         the_map.dog_sight_range, the_map.dog_sight_ang)

        self.right_bound = 0
        self.left_bound = 0
        self.within_bounds = False
        self.towards_right = True

        self.current_herd_middle_point = np.zeros(2)
        self.herding_velocity = np.zeros(2)


    def get_acceleration(self, obstacles, dt):
        """Finds the new acceleration"""
        obstacle = self.get_obstacle_agents(obstacles)
        o = self.obstacle_avoidance(obstacle)

        new_vel = self.vel + O*o + H * self.herding_velocity

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

    def set_herding_vel(self, right_bound, left_bound, herd_middle_point, radius, dt):
        """Sets the herding velocity for the dog. If inside of its bounds, goes towards the arc and end point,
        else goes towards the nearest bound """
        print("Radius: ", radius)

        self.right_bound = self.norm_ang(right_bound)
        self.left_bound = self.norm_ang(left_bound)
        self.current_herd_middle_point = herd_middle_point

        middle_dog_vec = (self.pos - self.current_herd_middle_point)
        dog_ang = self.norm_ang(atan2(middle_dog_vec[1], middle_dog_vec[0]))
        dog_dist_to_middle = np.linalg.norm(middle_dog_vec)

        bound_point_R = radius * np.array([cos(right_bound), sin(right_bound)]) + herd_middle_point
        bound_point_L = radius * np.array([cos(left_bound), sin(left_bound)]) + herd_middle_point


        self.within_bounds = self.is_withing_angles(left_bound, right_bound, dog_ang)

        if self.within_bounds:
            move_ang = self.norm_ang(self.give_angle((self.max_vel), radius))
            dir = 1
            if not self.towards_right:
                dir = -1

            target_ang = self.norm_ang(dog_ang + (dir * move_ang))
            target_point = radius * np.array([cos(target_ang), sin(target_ang)]) + herd_middle_point

            herding_vel = target_point - self.pos










        #    dir = 1
        #    if radius < dog_dist_to_middle:
        #        dir = -1
        #    #out_vel = (radius - dog_dist_to_middle) * (middle_dog_vec / dog_dist_to_middle)
        #    out_vel = dir * middle_dog_vec
        #    out_vel /= np.linalg.norm(out_vel)
        #    out_vel *= abs(radius - dog_dist_to_middle)
#
        #    if self.towards_right:
        #        dog_bound_point_vec = bound_point_R - self.pos
        #    else:
        #        dog_bound_point_vec = bound_point_L - self.pos
#
        #    dog_bound_point_vec /= np.linalg.norm(dog_bound_point_vec)
        #    #dog_bound_point_vec /= abs(radius - dog_dist_to_middle)
#
        #    herding_vel = out_vel + dog_bound_point_vec
        #    #herding_vel =  dog_bound_point_vec

        else:

            dog_bound_R_vec = bound_point_R - self.pos
            dog_bound_L_vec = bound_point_L - self.pos

            if np.linalg.norm(dog_bound_R_vec) < np.linalg.norm(dog_bound_L_vec):
                herding_vel = dog_bound_R_vec
            else:
                herding_vel = dog_bound_L_vec

        if np.linalg.norm(herding_vel) > self.max_vel:
            herding_vel /= np.linalg.norm(herding_vel)
            herding_vel *= self.max_vel

        self.herding_velocity = herding_vel

    def give_angle(self, arc_length, radius):
        angle = self.norm_ang(arc_length / radius)
        return angle

    def update(self, dt):
        super().update(dt)  # Updates the new position

        # Checks if the position is outside of the bounds
        middle_dog_vec = (self.pos - self.current_herd_middle_point)
        dog_ang = self.norm_ang(atan2(middle_dog_vec[1], middle_dog_vec[0]))
        within_angs = self.is_withing_angles(self.left_bound, self.right_bound, dog_ang)

        if self.within_bounds and not within_angs:
            self.towards_right = not self.towards_right

    def norm_ang(self, angle):
        return angle % (2*pi)






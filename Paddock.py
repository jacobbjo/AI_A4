from math import *
import numpy as np


from importJSON import Map
from Sheep import Sheep

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)



class Paddock:
    def __init__(self, map, square_size):
        self.map = map
        self.square_size = square_size
        self.all_sheep = []

        # Create grid, [0][0] is in SW corner of map
        self.grid_rows = int(ceil((self.map.bounding_polygon.y_max - self.map.bounding_polygon.y_min)
                              / self.square_size))
        self.grid_cols = int(ceil((self.map.bounding_polygon.x_max - self.map.bounding_polygon.x_min)
                              / self.square_size))
        self.grid = [[[] for c in range(self.grid_cols)] for r in range(self.grid_rows)]


    def generate_sheep(self):
        # does the initial placement of the sheep
        herd_start = self.map.herd_start_polygon

        for i in range(self.map.sheep_n):

            while True:
                x = herd_start.x_min + np.random.rand() * (herd_start.x_max - herd_start.x_min)
                y = herd_start.y_min + np.random.rand() * (herd_start.y_max - herd_start.y_min)
                new_sheep_pos = np.array([x, y])

                neighboring_squares = self.get_neighboring_squares(new_sheep_pos)

                correct_spawn = True

                if herd_start.contain_point(new_sheep_pos):
                    for current_square in neighboring_squares:
                        for sheep in current_square:
                            if np.linalg.norm(sheep.pos - new_sheep_pos) < self.map.sheep_r * 2:
                                correct_spawn = False
                else:
                    correct_spawn = False

                if correct_spawn:
                    new_sheep_vel = np.zeros(2)

                    if np.random.rand() < 0.3:
                        # only some sheep get an initial velocity
                        new_sheep_vel = np.random.normal(0, self.map.sheep_v_max/10, 2)


                    new_sheep = Sheep(new_sheep_pos, new_sheep_vel)
                    self.get_square(new_sheep_pos).append(new_sheep)
                    self.all_sheep.append(new_sheep)
                    break


    def get_square(self, position):
        """ Returns the grid square containing the given position"""
        grid_row = int((position[1] - self.map.bounding_polygon.y_min) // self.square_size)
        grid_col = int((position[0] - self.map.bounding_polygon.x_min) // self.square_size)
        return self.grid[grid_row][grid_col]

    def get_neighboring_squares(self, position):
        squares = []
        grid_row = int((position[1] - self.map.bounding_polygon.y_min) // self.square_size)
        grid_col = int((position[0] - self.map.bounding_polygon.x_min) // self.square_size)
        squares.append(self.grid[grid_row][grid_col])
        try:
            squares.append(self.grid[grid_row+1][grid_col-1])
        except:
            pass
        try:
            squares.append(self.grid[grid_row+1][grid_col])
        except:
            pass
        try:
            squares.append(self.grid[grid_row+1][grid_col+1])
        except:
            pass
        try:
            squares.append(self.grid[grid_row][grid_col+1])
        except:
            pass
        try:
            squares.append(self.grid[grid_row-1][grid_col+1])
        except:
            pass
        try:
            squares.append(self.grid[grid_row-1][grid_col])
        except:
            pass
        try:
            squares.append(self.grid[grid_row-1][grid_col-1])
        except:
            pass
        try:
            squares.append(self.grid[grid_row][grid_col-1])
        except:
            pass
        return squares


    def update_grid(self):
        self.grid = [[[] for c in range(self.grid_cols)] for r in range(self.grid_rows)]
        for sheep in self.all_sheep:
            self.get_square(sheep.pos).append(sheep)

    def get_neighbors_in_sight(self, current_sheep):
        neighbors = []

        squares = self.get_neighboring_squares(current_sheep.pos)

        for square in squares:
            for sheep in square:
                if self.in_range(current_sheep, sheep) and not sheep == current_sheep:
                    neighbors.append(sheep)

        return neighbors


    def in_range(self, sheep_a, sheep_b):
        """ Checks sheep_b is in range for sheep_a to care about it when moving """

        # The vector between sheep_a and sheep_b
        vec_ab = sheep_b.pos - sheep_a.pos

        if np.linalg.norm(vec_ab) < sheep_a.sight_range:
            sheep_a_ang = atan2(sheep_a.dir[1], sheep_a.dir[0])
            vec_ab_ang = atan2(vec_ab[1], vec_ab[0])  # The angle from the x-axis to vec_ab

            # Angles from the x-axis to the boundaries
            right_ang = sheep_a_ang - sheep_a.sight_ang/2
            left_ang = sheep_a_ang + sheep_a.sight_ang/2

            if right_ang < 0:
                right_ang += (2*np.pi)

            if left_ang < 0:
                left_ang += (2*np.pi)

            if vec_ab_ang < 0:
                vec_ab_ang += (2*np.pi)

            if right_ang > left_ang:
                # The velocity need to be larger than left and smaller than right
                return not left_ang < vec_ab_ang < right_ang

            # If/else to prevent from false negative when right_ang < left_ang < vel_ang
            if right_ang <= vec_ab_ang <= left_ang:
                return True
            else:
                return False
        else:
            return False



def plot_sheep(sheep, the_map):
    plt.clf()
    the_map.plot_map()

    for a_sheep in sheep:
        # Plots the position
        plt.plot(a_sheep.pos[0], a_sheep.pos[1], "o")

        # Plot velocity
        plt.plot([a_sheep.pos[0], a_sheep.vel[0] + a_sheep.pos[0]], [a_sheep.pos[1], a_sheep.vel[1] + a_sheep.pos[1]])
    plt.pause(0.05)



kart = Map("maps/M1.json")
padd = Paddock(kart, 5)
padd.generate_sheep()

kart.plot_map()
print(len(padd.grid))
print(len(padd.grid[0]))
print(padd.get_square(np.array([22, -10])))


#for i in range(600):
#    print(len(padd.grid))
#    print(i)
#    for sheep in padd.all_sheep:
#        neighbors = padd.get_neighbors_in_sight(sheep)
#        sheep.find_new_vel(neighbors, [], [])
#
#    for sheep in padd.all_sheep:
#        sheep.update()
#    plot_sheep(padd.all_sheep, kart)
#    padd.update_grid()
#
        #plt.plot(sheep.pos[0], sheep.pos[1], "ro")
        #plt.plot([sheep.pos[0], sheep.dir[0] + sheep.pos[0]], [sheep.pos[1], sheep.dir[1] + sheep.pos[1]])

#neighbors = padd.get_neighbors_in_sight(padd.all_sheep[30])

#plt.plot(padd.all_sheep[30].pos[0], padd.all_sheep[30].pos[1], "b*")

#for sheep in neighbors:
    #plt.plot(sheep.pos[0], sheep.pos[1], "c*")

#plt.show()


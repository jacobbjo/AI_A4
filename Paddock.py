from math import *
import numpy as np

from importJSON import Map
from Sheep import Sheep
from Dog import Dog

import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams['figure.figsize'] = (16, 9)


class Paddock:
    def __init__(self, map, square_size):
        self.map = map
        self.square_size = square_size
        self.all_sheep = []
        self.all_dogs = []

        # Create grid, [0][0] is in SW corner of map
        self.grid_rows = int(ceil((self.map.bounding_polygon.y_max - self.map.bounding_polygon.y_min)
                              / self.square_size))+1
        self.grid_cols = int(ceil((self.map.bounding_polygon.x_max - self.map.bounding_polygon.x_min)
                              / self.square_size))+1
        self.grid = [[[] for c in range(self.grid_cols)] for r in range(self.grid_rows)]
        self.obstacles_squares = self.place_obstacles()

        self.all_obstacles = self.map.obstacles[:]
        self.all_obstacles.append(self.map.bounding_polygon)
        self.sheep_middle_point = np.zeros(2)
        self.lost_sheep_pos = np.zeros(2)

    def place_obstacles(self):
        dictionary = {}
        # Adding the obstacles
        obstacles = self.map.obstacles[:]
        obstacles.append(self.map.bounding_polygon)
        for obstacle in obstacles:
            for i in range(len(obstacle.vertices)):
                p1 = obstacle.vertices[i]
                p2 = obstacle.vertices[i - 1]
                step = ((p1 - p2)/np.linalg.norm(p1 - p2)) * self.square_size

                current_point = p2.copy()

                while np.linalg.norm(p2 - current_point) < np.linalg.norm(p1 - p2):
                    square_ind = self.get_square_index(current_point)
                    if square_ind not in dictionary:
                        dictionary[square_ind] = [obstacle]
                    else:
                        if obstacle not in dictionary[square_ind]:
                            dictionary[square_ind].append(obstacle)
                    current_point += step

        return dictionary

    def generate_dogs(self):
        for dog_pos in self.map.dog_start_positions:
            self.all_dogs.append(Dog(self.map, dog_pos, np.array([-3.0, 3.0])))

    def generate_sheep(self):
        # does the initial placement of the sheep
        herd_start = self.map.herd_start_polygon
        tot_pos = np.zeros(2)

        for i in range(self.map.sheep_n):

            while True:
                x = herd_start.x_min + np.random.rand() * (herd_start.x_max - herd_start.x_min)/1
                y = herd_start.y_min + np.random.rand() * (herd_start.y_max - herd_start.y_min)/1
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

                    if np.random.rand() < 0.4:
                        # only some sheep get an initial velocity
                        new_sheep_vel = np.random.normal(0, self.map.sheep_v_max/10, 2)


                    new_sheep = Sheep(self.map, new_sheep_pos, new_sheep_vel)
                    self.get_square(new_sheep_pos).append(new_sheep)
                    self.all_sheep.append(new_sheep)
                    tot_pos += new_sheep_pos
                    break
        self.sheep_middle_point = tot_pos / len(self.all_sheep)
        hej = 1

    def get_square_index(self, position):
        """ returns the indices for the grid for the given position """
        grid_row = int((position[1] - self.map.bounding_polygon.y_min) // self.square_size)
        grid_col = int((position[0] - self.map.bounding_polygon.x_min) // self.square_size)

        if grid_row < 0 or grid_col < 0 or grid_row >= self.grid_rows or grid_col >= self.grid_cols:
            raise IndexError("Sheep is out of bounds")

        return grid_row, grid_col

    def get_square(self, position):
        """ Returns the grid square containing the given position"""
        ind = self.get_square_index(position)
        return self.grid[ind[0]][ind[1]]

    def get_neighbor_square_index(self, position):
        ok_indices = []
        ind = self.get_square_index(position)

        grid_row = ind[0]
        grid_col = ind[1]

        for i in range(-1, 2):
            for j in range(-1, 2):
                try:
                    self.grid[grid_row + i][grid_col + j]
                    ok_indices.append((grid_row + i, grid_col + j))
                except:
                    pass
        return ok_indices

    def get_neighboring_squares(self, position):
        squares = []

        neighbor_index = self.get_neighbor_square_index(position)

        for index in neighbor_index:
            squares.append(self.grid[index[0]][index[1]])
        return squares

    def update_grid(self):
        self.grid = [[[] for c in range(self.grid_cols)] for r in range(self.grid_rows)]
        for sheep in self.all_sheep:
            self.get_square(sheep.pos).append(sheep)

    def update(self):
        middle_point = np.zeros(2)
        for sheep in self.all_sheep:
            sheep.find_new_vel(padd.get_neighboring_squares(sheep.pos), padd.all_obstacles, self.all_dogs, self.map.dt)

        self.arc_following()
        for dog in self.all_dogs:
            dog.find_new_vel(self.all_obstacles, self.map.dt)

        for sheep in self.all_sheep:
            sheep.update(self.map.dt)
            middle_point += sheep.pos

        for dog in self.all_dogs:
            dog.update(self.map.dt)

        self.update_grid()
        self.sheep_middle_point = middle_point/len(self.all_sheep)

        # Finds the position of the sheep furthest away from the middle point
        currently_max_dist = 0
        current_max_pos = self.sheep_middle_point
        for sheep in self.all_sheep:
            dist_to_middle = np.linalg.norm(self.sheep_middle_point - sheep.pos)
            if dist_to_middle > currently_max_dist:
                currently_max_dist = dist_to_middle
                current_max_pos = sheep.pos
        self.lost_sheep_pos = current_max_pos

    def arc_following(self):
        #radius = np.linalg.norm(self.sheep_middle_point - self.lost_sheep_pos) * 1.1
        radius = 60
        fixed_middle_point = np.array([80, 0])
        #middle_line = self.sheep_middle_point - self.map.herd_goal_center
        middle_line = fixed_middle_point - self.map.herd_goal_center

        middle_line_ang = atan2(middle_line[1], middle_line[0])  # the angle to the x-axis from middle line

        dog_sector_ang = self.map.dog_chase_arc_ang / len(self.all_dogs)

        tot_right_bound = middle_line_ang + (self.map.dog_chase_arc_ang / 2)

        for ind, dog in enumerate(self.all_dogs):
            right_bound = tot_right_bound - (dog_sector_ang * ind)
            left_bound = tot_right_bound - (dog_sector_ang * (ind + 1))
            #dog.set_herding_vel(right_bound, left_bound, self.sheep_middle_point, radius)
            dog.set_herding_vel(right_bound, left_bound, fixed_middle_point, radius)





def give_angle(arc_length, radius):
    angle = arc_length/radius
    return angle


def animate(i):
    print("i:", i)

    for ind, animal in enumerate(all_animals):
        animate_objects[ind].set_xdata(animal.pos[0])
        animate_objects[ind].set_ydata(animal.pos[1])
        animate_objects[ind+len(all_animals)].set_xdata([animal.pos[0], animal.dir[0] + animal.pos[0]])
        animate_objects[ind+len(all_animals)].set_ydata([animal.pos[1], animal.dir[1] + animal.pos[1]])
        animate_objects[ind+2*len(all_animals)].set_xdata([animal.pos[0], animal.vel[0] + animal.pos[0]])
        animate_objects[ind+2*len(all_animals)].set_ydata([animal.pos[1], animal.vel[1] + animal.pos[1]])
    animate_objects[-2].set_xdata(padd.sheep_middle_point[0])
    animate_objects[-2].set_ydata(padd.sheep_middle_point[1])
    animate_objects[-1].set_xdata(padd.lost_sheep_pos[0])
    animate_objects[-1].set_ydata(padd.lost_sheep_pos[1])


    padd.update()

    return animate_objects


kart = Map("maps/M2.json")
padd = Paddock(kart, 7)
padd.generate_sheep()
padd.generate_dogs()

# Creates the plot objects
figure = kart.plot_map()

# The sheep and dog dots
plot_sheep = [plt.plot(sheep.pos[0], sheep.pos[1], "bo",  animated=True)[0] for sheep in padd.all_sheep]
plot_dogs = [plt.plot(dog.pos[0], dog.pos[1], "o",  animated=True)[0] for dog in padd.all_dogs]

# The direction for the sheep and dogs
sheep_dir = [plt.plot([sheep.pos[0], sheep.dir[0] + sheep.pos[0]], [sheep.pos[1], sheep.dir[1] + sheep.pos[1]], "b",
                      animated=True)[0] for sheep in padd.all_sheep]
dog_dir = [plt.plot([dog.pos[0], dog.dir[0] + dog.pos[0]], [dog.pos[1], dog.dir[1] + dog.pos[1]], "r", animated=True)[0]
           for dog in padd.all_dogs]

# The velocity for the sheep and dogs
sheep_vel = [plt.plot([sheep.pos[0], sheep.vel[0] + sheep.pos[0]], [sheep.pos[1], sheep.vel[1] + sheep.pos[1]], "g",
                      animated=True)[0] for sheep in padd.all_sheep]
dog_vel = [plt.plot([dog.pos[0], dog.vel[0] + dog.pos[0]], [dog.pos[1], dog.vel[1] + dog.pos[1]], "g", animated=True)[0]
           for dog in padd.all_dogs]

middle_point_plot = plt.plot(padd.sheep_middle_point[0], padd.sheep_middle_point[1], "*g", animated=True)[0]
lost_sheep_plot = plt.plot(padd.lost_sheep_pos[0], padd.lost_sheep_pos[1], "oc", animated=True)[0]

# Plots circle for debuging

animate_objects = plot_sheep + plot_dogs + sheep_dir + dog_dir + sheep_vel + dog_vel
animate_objects.append(middle_point_plot)
animate_objects.append(lost_sheep_plot)
all_animals = padd.all_sheep + padd.all_dogs
all_dirs = sheep_dir + dog_dir


ani = animation.FuncAnimation(figure, animate, interval=0, blit=True)
plt.show()


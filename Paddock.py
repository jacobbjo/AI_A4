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
                    break

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

    def get_obstacle_agents(self, current_sheep):

        def find_intersecting_point(path_line, obs_edge):
            """ Finds the point where the lines cross. The lines are np.arrays of np-arrays"""
            x_mat_line1, x_mat_line2, y_mat_line1, y_mat_line2 = np.ones(path_line.shape), np.ones(path_line.shape), \
                                                                 np.ones(path_line.shape), np.ones(path_line.shape)
            x_mat_line1[:, 0] = path_line[:, 0]
            x_mat_line2[:, 0] = obs_edge[:, 0]
            y_mat_line1[:, 0] = path_line[:, 1]
            y_mat_line2[:, 0] = obs_edge[:, 1]

            div_det = np.linalg.det(
                np.array([[np.linalg.det(x_mat_line1), np.linalg.det(y_mat_line1)], [np.linalg.det(x_mat_line2),
                                                                                     np.linalg.det(y_mat_line2)]]))
            x_det = np.linalg.det(
                np.array([[np.linalg.det(path_line), np.linalg.det(x_mat_line1)], [np.linalg.det(obs_edge),
                                                                                   np.linalg.det(x_mat_line2)]]))
            y_det = np.linalg.det(
                np.array([[np.linalg.det(path_line), np.linalg.det(y_mat_line1)], [np.linalg.det(obs_edge),
                                                                                   np.linalg.det(y_mat_line2)]]))
            return np.array([x_det / div_det, y_det / div_det])

        near_obstacles = []

        obstacles = self.map.obstacles[:]
        obstacles.append(self.map.bounding_polygon)

        # The line to compare with the obstacle edge
        sheep_collision_line = np.array([current_sheep.pos, current_sheep.pos + current_sheep.dir * self.map.sheep_sight_range*8])
        point_diff = sheep_collision_line[1] - sheep_collision_line[0]

        normal = np.array([-point_diff[1], point_diff[0]])
        normal /= np.linalg.norm(normal)
        normal *= np.linalg.norm(current_sheep.radius)

        sheep_collision_line1 = sheep_collision_line + normal
        sheep_collision_line2 = sheep_collision_line - normal

        # Finds the closest point of intersection from the sheep to each obstacle
        for obstacle in obstacles:
            min_intersect_point = None
            min_intersect_point_dist = float("infinity")
            best_vel = np.zeros(2)

            for i in range(len(obstacle.vertices)):
                obs_edge = np.array([obstacle.vertices[i], obstacle.vertices[i-1]])

                intersect_point1 = None
                intersect_point2 = None
                intersect_point = None

                if obstacle.lines_intersect(obs_edge[0], obs_edge[1], sheep_collision_line1[0], sheep_collision_line1[1]):
                    intersect_point1 = find_intersecting_point(sheep_collision_line1, obs_edge)

                if obstacle.lines_intersect(obs_edge[0], obs_edge[1], sheep_collision_line2[0], sheep_collision_line2[1]):
                    intersect_point2 = find_intersecting_point(sheep_collision_line2, obs_edge)

                if intersect_point1 is not None:
                    intersect_point = intersect_point1

                if intersect_point2 is not None and intersect_point is not None:
                    # Check the closest
                    if np.linalg.norm(current_sheep.pos - intersect_point2) < \
                            np.linalg.norm(current_sheep.pos - intersect_point):
                        intersect_point = intersect_point2
                else:
                    intersect_point = intersect_point2

                if intersect_point is not None:
                    if np.linalg.norm(intersect_point - current_sheep.pos) < self.map.sheep_sight_range*7:
                        if np.linalg.norm(intersect_point - current_sheep.pos) < min_intersect_point_dist:

                            point_diff = obs_edge[1] - obs_edge[0]

                            normal1 = np.array([-point_diff[1], point_diff[0]])
                            normal1 /= np.linalg.norm(normal1)
                            normal1 *= np.linalg.norm(current_sheep.vel)

                            normal2 = normal1 * -1

                            #vel1 = ((obs_edge[0] - obs_edge[1])/np.linalg.norm(obs_edge[0] - obs_edge[1])) * \
                            #       np.linalg.norm(current_sheep.vel)
                            #vel2 = -1 * vel1

                            diff_vel1 = np.linalg.norm(current_sheep.vel - normal1)
                            diff_vel2 = np.linalg.norm(current_sheep.vel - normal2)

                            if diff_vel1 > diff_vel2:
                                velocity = normal1
                            else:
                                velocity = normal2

                            min_intersect_point = intersect_point
                            best_vel = velocity
                            min_intersect_point_dist = np.linalg.norm(intersect_point - current_sheep.pos)

            if min_intersect_point is not None:
                near_obstacles.append(Sheep(self.map, min_intersect_point, best_vel))

        close_sheep = None
        min_dist_to_obstacle = float("infinity")
        for obs_sheep in near_obstacles:
            dist = np.linalg.norm(current_sheep.pos - obs_sheep.pos)
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist
                close_sheep = obs_sheep



        return close_sheep

    def update_grid(self):
        self.grid = [[[] for c in range(self.grid_cols)] for r in range(self.grid_rows)]
        for sheep in self.all_sheep:
            self.get_square(sheep.pos).append(sheep)

    def get_neighbors_in_sight(self, current_sheep):
        neighbors = []
        squares = self.get_neighboring_squares(current_sheep.pos)

        for square in squares:
            for sheep in square:


                #if np.linalg.norm(sheep.pos - current_sheep.pos) < self.map.sheep_r*2  and not sheep == current_sheep:
                #    print("SHEEP COLLISION")

                if self.in_range(current_sheep, sheep) and not sheep == current_sheep:
                    neighbors.append(sheep)

        return neighbors


    def in_range(self, sheep_a, sheep_b):
        """ Checks sheep_b is in range for sheep_a to care about it when moving """
        sight_range = self.map.sheep_sight_range
        sight_ang = self.map.sheep_sight_ang

        # The vector between sheep_a and sheep_b
        vec_ab = sheep_b.pos - sheep_a.pos

        if np.linalg.norm(vec_ab) < sight_range:
            sheep_a_ang = atan2(sheep_a.dir[1], sheep_a.dir[0])
            vec_ab_ang = atan2(vec_ab[1], vec_ab[0])  # The angle from the x-axis to vec_ab

            # Angles from the x-axis to the boundaries
            right_ang = sheep_a_ang - sight_ang/2
            left_ang = sheep_a_ang + sight_ang/2

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

    def update(self):
        for sheep in self.all_sheep:
            neighbors = self.get_neighbors_in_sight(sheep)
            obstacles = self.get_obstacle_agents(sheep)
            sheep.find_new_vel(neighbors, obstacles, self.all_dogs, self.map.dt)

        for dog in self.all_dogs:
            dog.find_new_vel(self.all_obstacles, self.map.dt)

        for sheep in self.all_sheep:
            sheep.update(self.map.dt)

        for dog in self.all_dogs:
            dog.update(self.map.dt)

        self.update_grid()

def animate(i):
    print("i:", i)

    for ind, animal in enumerate(all_animals):
        animate_objects[ind].set_xdata(animal.pos[0])
        animate_objects[ind].set_ydata(animal.pos[1])
        animate_objects[ind+len(all_animals)].set_xdata([animal.pos[0], animal.dir[0] + animal.pos[0]])
        animate_objects[ind+len(all_animals)].set_ydata([animal.pos[1], animal.dir[1] + animal.pos[1]])
        animate_objects[ind+2*len(all_animals)].set_xdata([animal.pos[0], animal.vel[0] + animal.pos[0]])
        animate_objects[ind+2*len(all_animals)].set_ydata([animal.pos[1], animal.vel[1] + animal.pos[1]])

    padd.update()

    return animate_objects


kart = Map("maps/M1.json")
padd = Paddock(kart, 5)
padd.generate_sheep()
padd.generate_dogs()

# Creates the plot objects
figure = kart.plot_map()

# The sheep and dog dots
plot_sheep = [plt.plot(sheep.pos[0], sheep.pos[1], "bo",  animated=True)[0] for sheep in padd.all_sheep]
plot_dogs = [plt.plot(dog.pos[0], dog.pos[1], "ro",  animated=True)[0] for dog in padd.all_dogs]

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


animate_objects = plot_sheep + plot_dogs + sheep_dir + dog_dir + sheep_vel + dog_vel
all_animals = padd.all_sheep + padd.all_dogs
all_dirs = sheep_dir + dog_dir


ani = animation.FuncAnimation(figure, animate, interval=0, blit=True)
plt.show()


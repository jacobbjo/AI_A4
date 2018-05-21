import numpy as np
from math import *

class Animal:
    def __init__(self, pos, vel, radius, v_max, a_max, sight_range, sight_angle):
        self.pos = np.float64(pos)
        self.vel = vel
        self.radius = radius


        self.max_vel = v_max
        self.max_acc = a_max

        self.sight_range = sight_range
        self.sight_angle = sight_angle

        if np.linalg.norm(vel) == 0:
            rand_pos = np.random.normal(0.5, 0.5, 2)
            self.dir = rand_pos / np.linalg.norm(rand_pos)
        else:
            self.dir = vel / np.linalg.norm(vel)

        self.pos_hist = []
        self.next_vel = self.vel

    def update(self, the_map, dt):
        self.vel = self.next_vel
        if np.linalg.norm(self.vel) > 0.3:
            self.dir = self.vel / np.linalg.norm(self.vel)
        self.pos_hist.append(self.pos)
        new_pos = self.pos + (self.vel * dt)

        #map = the_map.bounding_polygon

        #if new_pos[0] < map.x_min:
        #    new_pos[0] = map.x_min + 0.1
        #if new_pos[1] < map.y_min:
        #    new_pos[1] = map.y_min + 0.1
        #if new_pos[0] > map.x_max:
        #    new_pos[0] = map.x_max - 0.1
        #if new_pos[1] > map.y_max:
        #    new_pos[1] = map.y_max - 0.1

        self.pos = new_pos


    def set_new_vel(self, new_acc, dt):
        if np.linalg.norm(new_acc) > self.max_acc:
            # Scale the acc vector
            new_acc /= np.linalg.norm(new_acc)
            new_acc *= self.max_acc

        self.next_vel = self.vel + (new_acc * dt)
        #self.next_vel = self.get_acceleration(neighbors, dt)

        if np.linalg.norm(self.next_vel) > self.max_vel:
            self.next_vel /= np.linalg.norm(self.next_vel)
            self.next_vel *= self.max_vel

    def get_obstacle_agents(self, obstacles):
        """
        Creates the obstacle agents on the closest edge of the obstacle
        """

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

        near_obstacles_animals = []

        # The line to compare with the obstacle edge
        animal_collision_line = np.array([self.pos, self.pos + self.dir * self.sight_range*8])
        point_diff = animal_collision_line[1] - animal_collision_line[0]

        normal = np.array([-point_diff[1], point_diff[0]])
        normal /= np.linalg.norm(normal)
        normal *= np.linalg.norm(self.radius)  # The normal for offset the line to the cylinder lines

        animal_collision_line1 = animal_collision_line + normal
        animal_collision_line2 = animal_collision_line - normal

        # Finds the closest point of intersection from the animal to each obstacle
        for obstacle in obstacles:
            min_intersect_point = None
            min_intersect_point_dist = float("infinity")
            best_vel = np.zeros(2)

            for i in range(len(obstacle.vertices)):  # Go through the edges of the obstacles
                obs_edge = np.array([obstacle.vertices[i], obstacle.vertices[i-1]])

                intersect_point1 = None
                intersect_point2 = None
                intersect_point = None

                if obstacle.lines_intersect(obs_edge[0], obs_edge[1], animal_collision_line1[0], animal_collision_line1[1]):
                    intersect_point1 = find_intersecting_point(animal_collision_line1, obs_edge)

                if obstacle.lines_intersect(obs_edge[0], obs_edge[1], animal_collision_line2[0], animal_collision_line2[1]):
                    intersect_point2 = find_intersecting_point(animal_collision_line2, obs_edge)

                if intersect_point1 is not None:
                    intersect_point = intersect_point1

                if intersect_point2 is not None and intersect_point is not None:
                    # Check the closest of the cylinder line
                    if np.linalg.norm(self.pos - intersect_point2) < \
                            np.linalg.norm(self.pos - intersect_point):
                        intersect_point = intersect_point2
                elif intersect_point2 is not None:
                    intersect_point = intersect_point2

                if intersect_point is not None:  # One of the lines are crossing an obstacle
                    if np.linalg.norm(intersect_point - self.pos) < self.sight_range*7:
                        if np.linalg.norm(intersect_point - self.pos) < min_intersect_point_dist:

                            point_diff = obs_edge[1] - obs_edge[0]

                            normal1 = np.array([-point_diff[1], point_diff[0]])
                            normal1 /= np.linalg.norm(normal1)
                            normal1 *= np.linalg.norm(self.vel)

                            normal2 = normal1 * -1

                            diff_vel1 = np.linalg.norm(self.vel - normal1)
                            diff_vel2 = np.linalg.norm(self.vel - normal2)

                            if diff_vel1 > diff_vel2:
                                velocity = normal1
                            else:
                                velocity = normal2

                            min_intersect_point = intersect_point
                            best_vel = velocity
                            min_intersect_point_dist = np.linalg.norm(intersect_point - self.pos)

            if min_intersect_point is not None:
                near_obstacles_animals.append(Animal(min_intersect_point, best_vel*3, 0, 0, 0, 0, 0))


        close_animal = None
        min_dist_to_obstacle = float("infinity")
        for obs_animal in near_obstacles_animals:
            dist = np.linalg.norm(self.pos - obs_animal.pos)
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist
                close_animal = obs_animal

        return close_animal


    def get_obstacle_agents2(self, obstacles):
        # Kolla om något hinder är inom räckhåll
        # Skapa en lista med alla hinderkanter i ett
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

        obstacle_edges = []  # List with all the edges of the obstacles
        obstacle_agents = []
        sight_amplifier = 7

        for obstacle in obstacles:
            for i in range(len(obstacle.vertices)):  # Go through the edges of the obstacles
                obstacle_edges.append(np.array([obstacle.vertices[i], obstacle.vertices[i-1]]))

        # Finds the two lines from the agent towards the obstacle (the cylinder)
        agent_middle_path = np.array([self.pos, self.pos + self.dir * self.sight_range*100])
        point_diff = agent_middle_path[1] - agent_middle_path[0]

        normal = np.array([-point_diff[1], point_diff[0]])  # According to formula
        normal /= np.linalg.norm(normal)
        normal *= np.linalg.norm(self.radius)  # The normal for offset the line to the cylinder lines

        animal_collision_right = agent_middle_path + normal
        animal_collision_left = agent_middle_path - normal

        intersecting_points = []

        for edge in obstacle_edges:  # Goes through all edges and saves those that are intersecting with the path
            edge_normal = self.find_edge_normal(edge)

            if obstacles[0].lines_intersect(edge[0], edge[1], animal_collision_right[0], animal_collision_right[1]):
                found_intersecting_point = find_intersecting_point(animal_collision_right, edge)
                if np.linalg.norm(found_intersecting_point - self.pos) < sight_amplifier * self.sight_range:
                    intersecting_points.append(found_intersecting_point)
            if obstacles[0].lines_intersect(edge[0], edge[1], animal_collision_left[0], animal_collision_left[1]):
                found_intersecting_point = find_intersecting_point(animal_collision_left, edge)
                if np.linalg.norm(found_intersecting_point - self.pos) < sight_amplifier * self.sight_range:
                    intersecting_points.append(found_intersecting_point)

            edge_line_parameters = self.find_line_eq(edge[0], edge[1])
            dist, point_on_obstacle = self.dist_point_to_line(self.pos, edge_line_parameters)
            if dist < self.sight_range *  sight_amplifier:
                vel = edge_normal / np.linalg.norm(edge_normal)
                vel *= self.max_vel
                obstacle_agents.append(Animal(point_on_obstacle, vel, self.radius, 0, 0, 0, 0))

        closest_distance = float("infinity")
        closest_intersecting_point = None

        for intersecting_point in intersecting_points:
            if np.linalg.norm(intersecting_point - self.pos) < closest_distance:
                closest_intersecting_point = intersecting_point
                closest_distance = np.linalg.norm(intersecting_point - self.pos)

        # The closest point on the course
        if closest_intersecting_point is not None:
            obstacle_agents.append(Animal(self.pos, -1*self.vel, self.radius, 0, 0, 0, 0))

        return obstacle_agents

    def find_edge_normal(self, edge):
        edge_point_diff = edge[1] - edge[0]
        edge_normal = np.array([-edge_point_diff[1], edge_point_diff[0]])
        compare_point = edge[0] + edge_normal
        opposite_point = edge[0] + (edge_normal * -1)

        if np.linalg.norm(compare_point - self.pos) < np.linalg.norm(opposite_point - self.pos):
            return edge_normal / np.linalg.norm(edge_normal)
        else:
            return (edge_normal / np.linalg.norm(edge_normal)) * -1

    def is_withing_angles(self, ang_1, ang_2, vec_ang):

        if ang_1 < 0:
            ang_1 += (2 * np.pi)

        if ang_2 < 0:
            ang_2 += (2 * np.pi)

        if vec_ang < 0:
            vec_ang += (2 * np.pi)

        if ang_1 > ang_2:
            # The velocity need to be larger than left and smaller than right

            return not ((ang_2 < vec_ang) and (vec_ang < ang_1))

        # If/else to prevent from false negative when right_ang < left_ang < vel_ang
        if ang_1 <= vec_ang <= ang_2:
            return True
        else:
            return False

    def find_line_eq(self, point1, point2):
        """Define a linear equation of the form ax + by + c = 0. Returns the a, b and c on the form (a, b, c)"""
        a = point1[1] - point2[1]
        b = point2[0] - point1[0]
        c = point1[0] * point2[1] - point2[0]*point1[1]

        return (a, b, c)

    def dist_point_to_line(self, point, line_parameters):
        """Computes the distance from a point to a line. Returns the distance and the closest point on the line"""
        a = line_parameters[0]
        b = line_parameters[1]
        distance = abs(a*point[0] + b*point[1] + line_parameters[2])/np.sqrt(a**2 + b ** 2)
        close_x = (b * (b*point[0] - a*point[1]) - a*line_parameters[2])/(a**2 + b**2)
        close_y = (a * (-b*point[0] + a*point[1]) - b*line_parameters[2])/(a**2 + b**2)

        return distance, np.array([close_x, close_y])


    def get_neighbors_in_sight(self, neighboring_squares):
        """ Finds the neighbors in range """
        neighbors = []

        for square in neighboring_squares:
            for sheep in square:

                #if np.linalg.norm(sheep.pos - current_sheep.pos) < self.map.sheep_r*2  and not sheep == current_sheep:
                #    print("SHEEP COLLISION")

                if self.in_range(sheep) and not sheep == self:
                    neighbors.append(sheep)

        return neighbors

    def in_range(self, sheep_b):
        """ Checks sheep_b is in range for sheep_a to care about it when moving """
        sight_range = self.sight_range
        sight_ang = self.sight_angle

        # The vector between sheep_a and sheep_b
        vec_ab = sheep_b.pos - self.pos

        if np.linalg.norm(vec_ab) < sight_range:

            sheep_a_ang = atan2(self.dir[1], self.dir[0])
            vec_ab_ang = atan2(vec_ab[1], vec_ab[0])  # The angle from the x-axis to vec_ab

            # Angles from the x-axis to the boundaries
            right_ang = sheep_a_ang - sight_ang/2
            left_ang = sheep_a_ang + sight_ang/2

            return self.is_withing_angles(right_ang, left_ang, vec_ab_ang)

        else:
            return False

import numpy as np

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

        map = the_map.bounding_polygon

        if new_pos[0] < map.x_min:
            new_pos[0] = map.x_min + 0.1
        if new_pos[1] < map.y_min:
            new_pos[1] = map.y_min + 0.1
        if new_pos[0] > map.x_max:
            new_pos[0] = map.x_max - 0.1
        if new_pos[1] > map.y_max:
            new_pos[1] = map.y_max - 0.1

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
        normal *= np.linalg.norm(self.radius)

        animal_collision_line1 = animal_collision_line + normal
        animal_collision_line2 = animal_collision_line - normal

        # Finds the closest point of intersection from the animal to each obstacle
        for obstacle in obstacles:
            min_intersect_point = None
            min_intersect_point_dist = float("infinity")
            best_vel = np.zeros(2)

            for i in range(len(obstacle.vertices)):
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
                    # Check the closest
                    if np.linalg.norm(self.pos - intersect_point2) < \
                            np.linalg.norm(self.pos - intersect_point):
                        intersect_point = intersect_point2
                else:
                    intersect_point = intersect_point2

                if intersect_point is not None:
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
                near_obstacles_animals.append(Animal(min_intersect_point, best_vel*2, 0, 0, 0, 0, 0))

        close_animal = None
        min_dist_to_obstacle = float("infinity")
        for obs_animal in near_obstacles_animals:
            dist = np.linalg.norm(self.pos - obs_animal.pos)
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist
                close_animal = obs_animal

        return close_animal


    def is_withing_angles(self, ang_1, ang_2, vec_ang):

        if ang_1 < 0:
            ang_1 += (2 * np.pi)

        if ang_2 < 0:
            ang_2 += (2 * np.pi)

        if vec_ang < 0:
            vec_ang += (2 * np.pi)

        if ang_1 > ang_2:
            # The velocity need to be larger than left and smaller than right
            return not ang_2 < vec_ang < ang_1

        # If/else to prevent from false negative when right_ang < left_ang < vel_ang
        if ang_1 <= vec_ang <= ang_2:
            return True
        else:
            return False


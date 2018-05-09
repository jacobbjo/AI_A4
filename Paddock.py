import math
import numpy as np

from importJSON import Map


class Paddock:
    def __init__(self, map, square_size):
        self.map = map
        self.square_size = square_size
        self.all_sheep = []

        # Create grid, [0][0] is in SW corner of map
        grid_rows = math.ceil((self.map.bounding_polygon.y_max - self.map.bounding_polygon.y_min)
                              / self.square_size)
        grid_cols = math.ceil((self.map.bounding_polygon.x_max - self.map.bounding_polygon.x_min)
                              / self.square_size)
        self.grid = [[[] for c in range(grid_cols)] for r in range(grid_rows)]


    def place_sheep(self):
        # does the initial placement of the sheep
        num_sheep = self.map.sheep_n
        herd_start_pos = self.map.herd_start_position

        # all_sheep.append(created_sheep)
        return

    def get_square(self, position):
        """ Returns the grid square containing the given position"""
        grid_row = (position[1] - self.map.bounding_polygon.y_min) // self.square_size
        grid_col = (position[0] - self.map.bounding_polygon.x_min) // self.square_size
        return self.grid[grid_row][grid_col]



kart = Map("maps/M1.json")
padd = Paddock(kart, 5)

print(len(padd.grid))
#print(padd.grid)

pos = np.array([-2, -12])

padd.get_square(pos).append("bajhs")

for row in padd.grid:
    print(row)

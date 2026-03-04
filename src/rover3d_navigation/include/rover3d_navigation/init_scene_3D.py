from matplotlib.patches import Polygon
import trimesh
import numpy as np


class initscene:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height


class obstacle:  # Generate obstacles
    def __init__(self, vertices):
        self.vertices = vertices
        self.polyhedron = trimesh.convex.convex_hull(vertices)


class ObstacleManager:
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def is_colliding(self, agent_x, agent_y, agent_z):
        point = np.array([agent_x, agent_y, agent_z])
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return True
        return False

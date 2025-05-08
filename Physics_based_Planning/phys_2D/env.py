"""
Environment for phys_2D
@author: James Pennington
@credit: huiming zhou
"""


class Env:
    def __init__(self, variant=None):
        if variant == 0 or variant is None:
            self.x_range = (0, 50)
            self.y_range = (0, 30)
        elif variant == 1: 
            self.x_range = (0, 50)
            self.y_range = (0, 30)
        elif variant == 2:
            self.x_range = (0, 49)
            self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary(variant)
        self.obs_circle = self.obs_circle(variant)
        self.obs_rectangle = self.obs_rectangle(variant)

    @staticmethod
    def obs_boundary(variant=None):
        if variant == 0 or variant is None:
            obs_boundary = [
                [0, 0, 1, 30],
                [0, 30, 50, 1],
                [1, 0, 50, 1],
                [50, 1, 1, 30]
            ]
        elif variant == 1:
            obs_boundary = [
                [0, 0, 1, 30],
                [0, 30, 50, 1],
                [1, 0, 50, 1],
                [50, 1, 1, 30]
            ]
        elif variant == 2:
            obs_boundary = [
                [0, 0, 50, 1],
                [0, 1, 1, 30],
                [0, 31, 50, 1],
                [50, 0, 1, 32]
            ]
        return obs_boundary

    @staticmethod
    def obs_rectangle(variant=None):
        if variant == 0 or variant is None:
            obs_rectangle = [
                [14, 12, 8, 2],
                [18, 22, 8, 3],
                [26, 7, 2, 12],
                [32, 14, 10, 2]
            ]
        elif variant == 1:
            obs_rectangle = [
                [13, 6, 26, 2],
                [37, 8, 2, 16],
                [13, 11, 4, 10],
                [13, 24, 26, 2]
            ]
        elif variant == 2:
            obs_rectangle = [
                [5, 5, 1, 26],
                [10, 1, 1, 26],
                [15, 5, 1, 26],
                [20, 1, 1, 26],
                [25, 5, 1, 26],
                [30, 1, 1, 26],
                [35, 5, 1, 26],
                [40, 1, 1, 26],
                [45, 5, 1, 26]
            ]

        return obs_rectangle

    @staticmethod
    def obs_circle(variant=None):
        if variant == 0 or variant is None:
            obs_cir = [
                [7, 12, 3],
                [46, 20, 2],
                [15, 5, 2],
                [37, 7, 3],
                [37, 23, 3]
            ]
        elif variant == 1:
            obs_cir = [
                [7, 8, 2],
                [7, 24, 2],
                [28, 16, 5],
                [45, 7, 3],
                [45, 25, 3]
            ]
        elif variant == 2:
            obs_cir = []

        return obs_cir

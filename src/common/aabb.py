import numpy as np


class AABB:

    def __init__(self, a, b):
        self.min = np.array(a)
        self.max = np.array(b)

    def inflate(self, amount):
        self.min -= amount
        self.max += amount
import unittest
import numpy as np
from classes.compactmatrix import *


class CompactMatrixTest(unittest.TestCase):

    def __init__(self, tests=()):
        super(CompactMatrixTest, self).__init__(tests)
        self.matrix = None
        self.size = (10, 10)
        self.full_mat = np.random.rand(self.size[0], self.size[1])

    def setUp(self):
        pass

    def test_construction(self):
        self.matrix = CompactMatrix(self.full_mat)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                self.assertEqual(self.matrix.matrix[row][col][0], row, "Constructor error : row")
                self.assertEqual(self.matrix.matrix[row][col][1], col, "Constructor error : col")
                self.assertEqual(self.matrix.matrix[row][col][2], self.full_mat[row, col], "Constructor error : value")

    def test_get(self):
        self.matrix = CompactMatrix(self.full_mat)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                self.assertEqual(self.full_mat[row, col], self.matrix[row, col], "Getter error : tuple indexing")
                self.assertEqual(self.full_mat[row, col], self.matrix[row][col][2], "Getter error : scalar indexing")

    def test_set(self):
        self.matrix = CompactMatrix(self.full_mat)
        new = np.random.rand()
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                self.matrix[row, col] = new
                self.assertEqual(self.matrix[row, col], new, "Setter error : data doesn't match")

    def test_add(self):
        self.matrix = CompactMatrix(self.full_mat)
        new = np.random.rand()
        self.matrix.add(new, 0, 1)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                result = self.full_mat[row, col] + new
                result = 1 if result > 1 else result
                result = 0 if result < 0 else result
                self.assertEqual(self.matrix[row, col], result, "Add error : data doesn't match")

    def test_get_kernel(self):
        self.matrix = CompactMatrix(self.full_mat)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                kernel = self.matrix.get_kernel((row, col), self.size[0], (3, 3))
                self.assertEqual(kernel.shape, (3, 3), "Kernel error : size doesn't match")

    def test_get_all_weights(self):
        self.matrix = CompactMatrix(self.full_mat)
        weights = self.matrix.get_all_weights()
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                self.assertEqual(weights[col + 10 * row], self.full_mat[row, col],
                                 "Get all weigths error : data doesn't match")


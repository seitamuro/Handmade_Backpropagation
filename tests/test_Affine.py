import unittest
from layers import Affine
import numpy as np

class TestAffineLayer(unittest.TestCase):
    def test_init(self):
        a = Affine(np.random.random(1, 1), np.random.random(1))


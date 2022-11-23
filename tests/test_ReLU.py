import numpy as np
import unittest
from layers import ReLU

class TestReLULayer(unittest.TestCase):
    def test_init(self):
        layer = ReLU()
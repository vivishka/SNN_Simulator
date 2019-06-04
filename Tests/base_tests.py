import unittest
import sys
sys.path.append("..")
from classes.base import *


class TestObject(SimulationObject):
    objects = []

    def __init__(self):
        super(TestObject, self).__init__()
        TestObject.objects.append(self)


class SimulationObjectTest(unittest.TestCase):

    """ Runs tests about Base.py classes"""

    def setUp(self):
        pass

    def test_flush(self):
        obj = TestObject()
        TestObject.flush()
        self.assertEqual(TestObject.objects, [], "Flush didn't empty objects list")

    def test_helper_reset(self):
        Helper.reset()
        self.assertEqual(Helper.step_nb, 0, "Step number bad reset")
        self.assertEqual(Helper.time, 0., "Simulation time bad reset")
        self.assertEqual(Helper.nb, 0., "Step total number bad reset")
        self.assertEqual(Helper.input_index, 0, "Input index bad reset")
        self.assertEqual(Helper.input_period, 0, "Input period bad reset")
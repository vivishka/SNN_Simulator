
import sys
sys.dont_write_bytecode = True


class SimulationObject(object):
    """docstring for CustomObject."""

    # objects = {}

    # TODO:  __repr__
    def __init__(self, label=''):
        super(SimulationObject, self).__init__()
        self.label = label

    @classmethod
    def get_objects(cls):
        return cls.objects
        # TODO: perhaps return a copy and empty this list of multiple networks

    def step(self):
        pass


class Helper(object):
    time = 0
    dt = 0

    def __init__(self):
        pass

    @staticmethod
    def step():
        Helper.time += Helper.dt

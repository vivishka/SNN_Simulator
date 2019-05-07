
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

    def step(self):
        pass


class Helper(object):
    step_nb = 0
    time = 0
    dt = 0
    nb = 0

    def __init__(self):
        pass

    @staticmethod
    def step():
        Helper.time += Helper.dt
        Helper.step_nb += 1

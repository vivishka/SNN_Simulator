
import sys
sys.dont_write_bytecode = True


class SimulationObject(object):
    """docstring for CustomObject."""

    # objects = {}

    # TODO:  __repr__
    def __init__(self, label=''):
        super(SimulationObject, self).__init__()
        self.label = label
        self.time = 0

    @classmethod
    def get_objects(cls):
        return cls.objects
        # TODO: perhaps return a copy and empty this list of multiple networks

    # @classmethod
    # def addInstance(cls, cls_name, obj):
    #     # if cls.objects
    #     # TODO: maybe

    def step(self, dt):
        self.time += dt

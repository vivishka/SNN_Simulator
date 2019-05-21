import numpy as np
import logging as log
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

    logged_modules = []  # Helper, Neuron, Encoder, Connection, Simulator, Layer, All

    def __init__(self):
        pass

    @staticmethod
    def step():
        Helper.time += Helper.dt
        Helper.step_nb += 1

    @staticmethod
    def init_weight():
        return np.random.rand()

    @staticmethod
    def init_logging(filename, level, modules):
        log.basicConfig(filename=filename, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
        Helper.log('Helper', log.INFO, 'logging started with parameters ' + filename + ' w ' + '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        Helper.logged_modules = modules

    @staticmethod
    def log(module, level, message):
        if module in Helper.logged_modules or 'All' in Helper.logged_modules:
            if level == log.DEBUG:
                log.debug('simulation time: {0} - {1}: {2}'.format(Helper.time, module, message))
            if level == log.INFO:
                log.info('simulation time: {0} - {1}: {2}'.format(Helper.time, module, message))
            if level == log.WARNING:
                log.warning('simulation time: {0} - {1}: {2}'.format(Helper.time, module, message))
            if level == log.ERROR:
                log.error('simulation time: {0} - {1}: {2}'.format(Helper.time, module, message))
            if level == log.CRITICAL:
                log.critical('simulation time: {0} - {1}: {2}'.format(Helper.time, module, message))



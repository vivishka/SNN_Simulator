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
    time = 0.
    dt = 0.
    nb = 0.

    logged_modules = []  # Helper, Neuron, Encoder, Decoder, Connection, Simulator, Layer, All

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
                log.debug('simulation time: {0:.6f} - {1}: {2}'.format(Helper.time, module, message))
            elif level == log.INFO:
                log.info('simulation time: {0:.6f} - {1}: {2}'.format(Helper.time, module, message))
            elif level == log.WARNING:
                log.warning('simulation time: {0:.6f} - {1}: {2}'.format(Helper.time, module, message))
            elif level == log.ERROR:
                log.error('simulation time: {0:.6f} - {1}: {2}'.format(Helper.time, module, message))
            elif level == log.CRITICAL:
                log.critical('simulation time: {0:.6f} - {1}: {2}'.format(Helper.time, module, message))

    @staticmethod
    def get_index_2d(index_1d, length):
        """ returns the (row, column) for a given index and length
        ex index = 42, length = 10 => row = 4, col = 2
        """
        return index_1d // length, index_1d % length

    @staticmethod
    def get_index_1d(index_2d, length):
        """ returns the index for a given (row, column) and length
        ex index = (4, 2), length = 10 => index_1D = 42
        """
        return index_2d[0] * length + index_2d[1]

import numpy as np
import logging as log
import sys
import time
import platform
sys.dont_write_bytecode = True


class SimulationObject(object):
    """
    parent object for the simulator elements
    """

    # objects = {}
    # TODO: finish or remove labels
    def __init__(self, label=''):
        super(SimulationObject, self).__init__()
        self.label = label
        self.sim = None

    @classmethod
    def get_objects(cls):
        """
        Obtain the last instanced objects
        :return: SimulationObjects belonging to the same class that were created after the last flush
        :rtype: list of SimulationObject
        """
        return cls.objects

    def step(self):
        """
        Each SimulationObject has the step function
        To be overwritten
        """
        pass

    @classmethod
    def flush(cls):
        """ clear the list of SimulationObjects"""
        cls.objects = []


class Helper(object):

    logged_modules = []  # Helper, Neuron, Encoder, Decoder, Connection, Simulator, Layer, Learner, Dataset, All

    timings = {}
    nb_called = {}

    def __init__(self):
        pass

    @staticmethod
    def init_weight():
        return np.random.rand()

    @staticmethod
    def init_logging(filename, level, modules):
        """
        Start the logging
        :param filename: path to log file
        :type filename: str
        :param level: Only messages of equal or higher level wil be logged
        :type level: int
        :param modules: modules from which messages will be logged
        :type modules: list of str
        :return:
        """
        log.basicConfig(filename=filename, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
        Helper.log('Helper', log.INFO, 'logging started with parameters ' + filename + ' w ' + '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        Helper.logged_modules = modules

    @staticmethod
    def log(module, level, message):
        """
        log a message into the log file
        :param module: module the message is from. the message will not be logged if module not in logged_modules
        :type module:str
        :param level: only messages of equal or higher level wil be logged
        :type level: int, CRITICAL = 50, ERROR = 40, WARNING = 30, WARN = WARNING, INFO = 20, DEBUG = 10, NOTSET = 0
        :param message: message to be logged
        :type message: str
        :return:
        """
        if __name__ == '__main__':
            if module in Helper.logged_modules or 'All' in Helper.logged_modules:
                if level == log.DEBUG:
                    log.debug('{} {}'.format(module, message))
                elif level == log.INFO:
                    log.info('{} {}'.format(module, message))
                elif level == log.WARNING:
                    log.warning('{} {}'.format(module, message))
                elif level == log.ERROR:
                    log.error('{} {}'.format(module, message))
                elif level == log.CRITICAL:
                    log.critical('{} {}'.format(module, message))

    @staticmethod
    def get_index_2d(index_1d, length):
        """
        Transform 1D index into 2D index
        :return: the (row, column) for a given index and length
        ex index = 42, length = 10 => row = 4, col = 2
        :rtype: tuple
        """
        return index_1d // length, index_1d % length

    @staticmethod
    def get_index_1d(index_2d, length):
        """
        Transform 2D index into 1D index
        :return: the index for a given (row, column) and length
        ex index = (4, 2), length = 10 => index_1D = 42
        :rtype: int
        """
        return index_2d[0] * length + index_2d[1]

    @staticmethod
    def print_timings():
        """ Displays the timings of all the functions decorated by MeasureTiming"""
        for key in Helper.timings.keys():
            print("{}: {}, {}".format(key,  Helper.nb_called[key], Helper.timings[key]))

    @staticmethod
    def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        :param iteration: Required, current iteration
        :type iteration: int
        :param total: Required, total iterations
        :type total: int
        :param prefix: Optional, prefix string
        :type prefix: str
        :param suffix: Optional, suffix string
        :type suffix: str
        :param decimals: Optional, positive number of decimals in percent complete
        :type decimals: int
        :param bar_length: Optional, character length of bar
        :type bar_length: int
        """
        str_format = "{0:." + str(decimals) + "f}"
        # percents = str_format.format(100 * (iteration / float(total)))
        # filled_length = int(round(bar_length * iteration / float(total)))
        # bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        # sys.stdout.write('\x1b[2K\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))

        percents = f'{100 * (iteration / float(total)):.2f}'
        filled_length = int(round(bar_length * iteration / float(total)))
        if platform.system() == 'Windows':
            bar = f'{"█" * filled_length}{"▁" * (bar_length - filled_length)}'
        else:
            bar = f'{"o" * filled_length}{"_" * (bar_length - filled_length)}'

        sys.stdout.write(f'\r{prefix} |{bar}| {percents}% {suffix}'),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()


class MeasureTiming(object):
    """
    Decorator used to measure  the number of time a function is called and its total execution time
    """

    def __init__(self, name):
        self.name = name
        if name not in Helper.timings:
            Helper.timings[name] = 0
            Helper.nb_called[name] = 0

    def __call__(self, f):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """

        def wrapped_f(*args, **kwargs):
            start = time.time()
            r = f(*args, **kwargs)
            stop = time.time()
            Helper.timings[self.name] += (stop - start)
            Helper.nb_called[self.name] += 1
            return r

        return wrapped_f

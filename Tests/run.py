import unittest

from Tests.base_tests import *
from Tests.compactmatrix_tests import *
from Tests.connection_tests import *
class AllTestsSuite(unittest.TestSuite):

    def __init__(self):
        super(AllTestsSuite, self).__init__()
        self.addTest(SimulationObjectTest)
        self.addTest(CompactMatrixTest)
        self.addTest(ConnectionTest)
    if __name__ == '__main__':
        unittest.main()
import unittest
import numpy as np
from classes.connection import *
from classes.layer import *
from classes.neuron import *


class ConnectionTest(unittest.TestCase):

    def __init__(self, tests=()):
        super(ConnectionTest, self).__init__(tests)
        self.size = (10, 10)
        self.depth = 5

    def test_connection(self):
        layer_in = Ensemble(self.size, LIF())
        layer_out = Ensemble(self.size, LIF())
        bloc_in = Bloc(self.depth, self.size, LIF())
        bloc_out = Bloc(self.depth, self.size, LIF())

        con1 = Connection(layer_in, layer_out, 0, 1)
        con2 = Connection(bloc_in, layer_out, 0, 1)
        con3 = Connection(layer_in, bloc_out, 0, 1)
        con4 = Connection(bloc_in, bloc_out, 0, 1)

        self.assertEqual(con1.connection_list[0], con1)
        self.assertEqual(con1.source_e, layer_in)
        self.assertEqual(con1.dest_e, layer_out)
        self.assertEqual(Connection.objects[0], con1)
        self.assertEqual(con1.active, True)

        for index, con in enumerate(con2.connection_list):
            self.assertEqual(con.source_e, bloc_in.ensemble_list[index])
            self.assertEqual(con.dest_e, layer_out)
            self.assertEqual(con.active, True)
            self.assertIsNotNone(con.weights)

        self.assertEqual(len(con2.connection_list), self.depth)
        self.assertEqual(con2.dest_e, None)
        self.assertEqual(con2.active, False)

        for index, con in enumerate(con3.connection_list):
            self.assertEqual(con.dest_e, bloc_out.ensemble_list[index])
            self.assertEqual(con.source_e, layer_in)
            self.assertEqual(con.active, True)
            self.assertIsNotNone(con.weights)

        self.assertEqual(len(con3.connection_list), self.depth)
        self.assertEqual(con3.dest_e, None)
        self.assertEqual(con3.source_e, None)
        self.assertEqual(con3.active, False)

        for index, con in enumerate(con4.connection_list):
            self.assertIn(con.dest_e, bloc_out.ensemble_list)
            self.assertIn(con.source_e, bloc_in.ensemble_list)
            self.assertEqual(con.active, True)
            self.assertIsNotNone(con.weights)
        self.assertEqual(len(con4.connection_list), self.depth ** 2)
        self.assertEqual(con4.dest_e, None)
        self.assertEqual(con4.source_e, None)
        self.assertEqual(con4.active, False)

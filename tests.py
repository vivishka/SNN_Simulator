
import matplotlib.pyplot as plt
import numpy as np
from classes.network import Network
# from classes.neurons import Neuron
from classes.neuron import LIF
from classes.ensemble import Ensemble
from classes.simulator import Simulator
from classes.connection import Connection
from classes.probe import Probe
from classes.encoder import Node


def test_node(arg):
    n1 = Node(8, 0.3, 1.0)

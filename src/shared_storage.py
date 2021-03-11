import sys
sys.path.append("../src/")

import tensorflow as tf

from utilities import *
from config import *
from game import *
from replay_buffer import *
from networks import *

class SharedStorage(object):
    """Save the different versions of the network."""

    def __init__(self, network: SuperNetwork, uniform_network: UniformNetwork, optimizer: tf.keras.optimizers):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer

    def latest_network(self):
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.uniform_network

    def save_network(self, step: int, network: SuperNetwork):
        self._networks[step] = network

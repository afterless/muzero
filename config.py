import logging


class MuZeroConfig(object):
    def __init__(self,
                 training_steps=1000,
                 ):
        self.training_steps = training_steps

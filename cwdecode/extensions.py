#!/usr/bin/env python
#-*- coding: utf-8 -*-

import _pickle as cPickle

from blocks.roles import PARAMETER
from blocks.extensions import SimpleExtension
from blocks.filter import VariableFilter

def get_parameters(bricks):
    params = {}
    for brick in bricks:
        for parameter in brick.parameters:
            params[brick.name + "/" + parameter.name] = parameter
        children = brick.children
        if len(children) > 0:
            childparams = get_parameters(children)
            for parameter_name in childparams:
                params[brick.name + "/" + parameter_name] = childparams[parameter_name]
    return params

class SaveBestModel(SimpleExtension):
    def __init__(self, fname, bricks, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(SaveBestModel, self).__init__(**kwargs)
        self.bricks = bricks
        self.fname = fname

    def do(self, callback_name, *args):
        parameters = get_parameters(self.bricks)
        parameter_values = {}
        for parameter_name in parameters:
            parameter_values[parameter_name] = parameters[parameter_name].get_value()
        with open(self.fname, "wb") as f:
            cPickle.dump(parameter_values, f)

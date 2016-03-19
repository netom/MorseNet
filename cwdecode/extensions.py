#!/usr/bin/env python
#-*- coding: utf-8 -*-

from blocks.roles import PARAMETER
from blocks.extensions import SimpleExtension

class SaveBestModel(SimpleExtension):
    def __init__(self, **kwargs):
        kwargs.setdefault("after", True)
        super(SaveModel, self).__init__(**kwargs)
        self.path = path
        self.parameters = parameters
        self.save_separately = save_separately
        self.save_main_loop = save_main_loop
        self.use_cpickle = use_cpickle

    def do(self, callback_name, *args):
        _, from_user = self.parse_args(callback_name, args)

var_filter = VariableFilter(roles=[PARAMETER], bricks=[second_layer])
print(var_filter(cg.variables))

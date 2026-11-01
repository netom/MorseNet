#!/usr/bin/env python3

import tensorflow as tf
import yaml

from config.audio import *
from config.morse import *
from config.train import *

class ModelLoader(yaml.SafeLoader):
    pass

def variable_constructor(loader, node):
    name = loader.construct_scalar(node)
    return globals()[name]

ModelLoader.add_constructor("!var", variable_constructor)

def _build(o):
    """
    Build a tf.keras model described by Python dictionary.
    """
    if type(o) == list:
        return [ _build(x) for x in o ]
    if type(o) == dict:
        o_ = { k:_build(v) for k, v in filter(lambda t: len(t[0]) <= 0 or t[0][0] != "$", o.items()) }
        kind = o["$kind"]
        typ = o["$type"]
        if kind == "keras":
            f = getattr(tf.keras, typ)
            return f(**o_)
        if kind == "layer":
            f = getattr(tf.keras.layers, typ)
            return f(**o_)
        if kind == "regularizer":
            f = getattr(tf.keras.regularizers, typ)
            return f(**o_)
        raise ValueError(f"Invalid $kind: {kind}")
    return o

def build_model():
    with open("model.yaml", "r") as f:
        model_dict = yaml.load(f, ModelLoader)
    model = _build(model_dict["model"])
    model.run_eagerly = True
    model.build((None, TIMESTEPS, CHUNK))
    return model

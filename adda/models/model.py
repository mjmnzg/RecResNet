"""
PROJECT: Classifiers based on Deep Learning and Domain Adaptation for EEG
         signals from different subjects
File:   File to register model into execution.
AUTOR:  PhD Student. Magdiel Jiménez Guarneros
        email: magdiel.jg@inaoep.mx
        Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)
"""

models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        # set default parameters
        fn.range = None
        fn.mean = None
        fn.bgr = False
        return fn
    return decorator

def get_model_fn(name):
    return models[name]

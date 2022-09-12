import argparse

class Args(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

def get_default_args():
    args = {}
    args['task'] = 'classical_sr'
    args['training_patch_size'] = 64
    args['scale'] = 4
    args['noise'] = 15
    args['jpeg'] = 40
    args['large_model'] = False
    args['tile'] = None
    args['tile_overlap'] = None
    args['model_path'] = 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    initial_Args = Args(args)
    return initial_Args

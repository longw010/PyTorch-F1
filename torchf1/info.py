from .apputils.print_struct import info_str_tree, info_str_flat

__all__ = ['info_str', 'info_param']

def info_str(model, mode, input_size):
    if mode == 1:
        info_str_tree(model)
    elif mode == 2:
        # note channel first for the input size
        info_str_flat(model, input_size)
    return

def info_img(model):
    raise NotImplementedError

def info_param(model):
    return model.parameters()

def info_flop(model):
    raise NotImplementedError

def info_mac(model):
    raise NotImplementedError

if __name__ == '__main__':
    import torchvision.models as models
    alexnet = models.alexnet()
    print(info_param(alexnet))
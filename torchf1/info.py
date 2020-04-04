from utils.print_struct import info_str_tree, info_str_flat

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
    raise NotImplementedError

def info_flop(model):
    raise NotImplementedError

def info_mac(model):
    raise NotImplementedError

if __name__ == '__main__':
    import torchvision.models as models
    alexnet = models.alexnet()
    info_str(alexnet, 2, (3,224,224))
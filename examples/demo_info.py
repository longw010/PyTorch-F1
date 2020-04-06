from torchf1 import info_str, profile_macs
import torchvision.models as models
import torch


if __name__ == '__main__':
    alexnet = models.alexnet()
    # tree based ; similar to print (model)
    info_str(alexnet, 1, (3, 224, 224))
    # flat; similar to torchsummary
    info_str(alexnet, 2, (3, 224, 224))

    import numpy as np

    input_size = (1, 3, 64, 64)
    im_input = np.random.random(input_size)
    im_input = torch.from_numpy(im_input).float()
    im_out = alexnet(im_input)
    macs = profile_macs(alexnet, im_input)
    print (macs)


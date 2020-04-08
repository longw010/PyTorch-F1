"""
One click pruning
"""

import numpy as np
import torch.nn as nn

def prune_dry_run(model, gpu_id, input_size, batch_size):
    """
    Didn't run the pruning; just a checklist to do if you have GPU memory issue
    :return:
    """
    # TODO: detect the remaining GPU memory in the hardware
    get_usable_GPU(gpu_id)

    # estimate GPU needed based on current settings
    estimate_GPU(model, input_size, batch_size)

    # verify if the above is consistent

    # if the image shape is not changed for each inference, enable the cudnn
    print ("torch.backends.cudnn.benchmark = True")

    # try above; use separate func though
    # batch_size = 1
    print ('set batch_size = 1')

    # use FP16 (risky)
    print ('set data to FP16')

    # free the unused tensor
    print ('setting Relu(inplace=True)')

    # apply gradient accumulation
    print ("""
    for i, (features, target) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, target)  
        loss = loss / accumulation_steps

        loss.backward() 
        if ((i + 1) % accumulation_steps) == 0:
            optimizer.step() 
            optimizer.zero_grad() """)


def fp16_helper(model, input):
    # convert model to half precision
    model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    # iterate over data (normally used in the data loader)
    inputs = input.to('cuda').half()
    return model, inputs


def get_usable_GPU(gpu_id):
    raise NotImplemented


def estimate_GPU(model, input_size, batch_size):
    # GPU = model_params * n + batch_size * output_shape * 2 + input_field

    # TODO: get the parameters data type

    raise NotImplemented


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


if __name__ == '__main__':
    import torchvision.models as models

    alexnet = models.alexnet()
    prune_dry_run(alexnet, '-1')
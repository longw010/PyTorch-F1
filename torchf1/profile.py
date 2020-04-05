import torch
import numpy as np
import time

cuda = False
device = torch.device('cuda' if cuda else 'cpu')

def get_inf_time(model, input_size):
    if len(input_size) == 3:
        input_size = (1, input_size[0], input_size[1], input_size[2])

    # the input size must be 4
    assert len(input_size) == 4

    im_input = np.random.random(input_size)
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        # set GPU mode
        start = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
        end = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
        model = model.to(device)
        im_input = im_input.to(device)

        with torch.no_grad():
            start.record()
            model(im_input)
            end.record()
            # need to set synchronize way during profiling
            torch.cuda.synchronize()
            tic = start.elapsed_time(end)
    else:
        # set CPU mode with no CUDA enabled
        model = model.to(device)
        im_input = im_input.to(device)
        start_time = time.time()
        model(im_input)
        tic = time.time() - start_time

    return tic

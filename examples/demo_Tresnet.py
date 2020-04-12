
from torchf1 import info_str, profile_macs, profile_flops
import torchvision.models as models
import torch
from torchf1.sample_models.TResnet.Tresnet import TResnetM
from torchf1.profile import profile_macs, get_inf_time

if __name__ == '__main__':
    params = dict()
    params['num_classes'] = 3
    params['remove_aa_jit'] = False
    tresnet = TResnetM(params)
    input_size = (3, 64, 64)
    info_str(tresnet, 2, input_size)
    print(get_inf_time(tresnet, input_size))  # 0.025 for 3x64x64
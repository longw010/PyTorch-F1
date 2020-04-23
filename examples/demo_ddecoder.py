from torchf1 import info_str
import torchvision.models as models
from torchf1.sample_models.DeepDecoder.DDecoder import decodernw
import torch
from torchf1.profile import profile_macs, get_inf_time

if __name__ == '__main__':
    dd_model = decodernw()
    input_size = (1, 128, 12, 12)
    input_size = (128, 1, 1)
    info_str(dd_model, 2, input_size)


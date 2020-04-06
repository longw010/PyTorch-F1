from torchf1 import info_str, get_inf_time
import torchvision.models as models
from torchf1.sample_models.IMDN.IMDN import IMDNnet, IMDN_RTC, IMDN_AS
import torch
#import numpy as np
#from torchviz import make_dot

if __name__ == '__main__':
    imdn_model = IMDNnet()

    #im_input = np.random.random(input_size)
    #im_input = torch.from_numpy(im_input).float()

    #im_out = imdn_model(im_input)
    #make_dot(im_out).render("attached", format="png")

    input_size = (3, 64, 64) # 715,176 params; 2.73MB
    info_str(imdn_model, 2, input_size)
    #info_str(imdn_model, 1, (3, 64, 64))
    print (get_inf_time(imdn_model, input_size)) # 0.1265 for 3x64x64

    imdn_model = IMDN_RTC()
    info_str(imdn_model, 2, input_size)
    print(get_inf_time(imdn_model, input_size))  # 0.025 for 3x64x64

    imdn_model = IMDN_AS()
    info_str(imdn_model, 2, input_size)
    print(get_inf_time(imdn_model, input_size))  # 0.022 for 3x64x64
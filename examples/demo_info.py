from torchf1 import info_str
import torchvision.models as models

if __name__ == '__main__':
    alexnet = models.alexnet()
    # tree based ; similar to print (model)
    info_str(alexnet, 1)
    # flat; similar to torchsummary
    info_str(alexnet, (3, 224, 224))


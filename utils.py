import torch


def kaiming_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif "BatchNorm" in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


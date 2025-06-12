import torchvision.models as models
from core.models.patch_deconvnet import *
from core.models.section_deconvnet import *
from core.models.i2u_net import I2U_Net
from core.models.section_i2u_net import Section_I2U_Net

def get_model(name, pretrained, n_classes, dim=None):
    model = _get_model_instance(name)

    if name in ['section_deconvnet','patch_deconvnet']:
        model = model(n_classes=n_classes)
        if pretrained:
            try:
                from torchvision.models import VGG16_Weights
                vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except ImportError:
                vgg16 = models.vgg16(pretrained=pretrained)
        else:
            vgg16 = models.vgg16(pretrained=False)
        model.init_vgg16_params(vgg16)
    elif name == 'i2u_net':
        model = model(classes=n_classes, channels=1)
    elif name == 'section_i2u_net':
        model = model(classes=n_classes, channels=1)
    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'section_deconvnet': section_deconvnet,
            'patch_deconvnet': patch_deconvnet,
            'section_deconvnet_skip': section_deconvnet_skip,
            'patch_deconvnet_skip': patch_deconvnet_skip,
            'i2u_net': I2U_Net,
            'section_i2u_net': Section_I2U_Net,
        }[name]
    except:
        print(f'Model {name} not available')

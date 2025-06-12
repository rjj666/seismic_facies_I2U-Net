from core.loader.data_loader import *

def get_loader(arch):
    if 'patch' in arch: 
        return patch_loader
    elif 'section' in arch:
        return section_loader
    else:
        raise NotImplementedError()

class OriginalCustomSamplerTrain(torch.utils.data.Sampler):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __iter__(self):
        char = ['i' if np.random.randint(2) == 1 else 'x']
        self.indices = [idx for (idx, name) in enumerate(
            self.data_list) if char[0] in name]
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

class OriginalCustomSamplerVal(torch.utils.data.Sampler):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __iter__(self):
        char = ['i' if np.random.randint(2) == 1 else 'x']
        self.indices = [idx for (idx, name) in enumerate(
            self.data_list) if char[0] in name]
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
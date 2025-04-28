from torch.nn.parallel import DistributedDataParallel


class DDP(DistributedDataParallel):
    
    def get_param(self, name):
        return self.module.get_param(name)

    def get_module(self, name):
        return self.module.get_module(name)
import torch


class AE:
    def __init__(self, name):
        self.name = name

    def encode(self, x):
        """ We assume a channel-first input """
        raise NotImplementedError("Encoder")

    def decode(self, z):
        raise NotImplementedError("Decoder")

    def map_stack(self, x, func):
        if len(x.shape) == 1:
            return func(x)
        else:
            z = []
            for c in range(x.shape[0]):
                z.append(func(x[c]))
            return torch.stack(z)

    def autoencode(self, x):
        return self.decode(self.encode(x))


    def autoencode_multi(self, x, codec):
        raise NotImplemented("Multi-codec")
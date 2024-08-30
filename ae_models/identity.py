"""
Used to pass files through the same load/resample/save pipeline without changes
"""


from ae_models.ae import AE


class Identity(AE):
    def __init__(self):
        super().__init__("identity")

    def encode(self, x):
        return x

    def decode(self, x):
        return x

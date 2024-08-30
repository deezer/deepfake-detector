import json
import os

default_path = "./conf"

class ConfLoader():
    def __init__(self, path, base_name = "base"):
        self.path = path

        with open(os.path.join(self.path, base_name+'.json')) as f:
            self.base = json.load(f)
        self.model = {}
        self.self_update()

    def load_model(self, m_name):
        try:
            with open(os.path.join(self.path, m_name+'.json')) as f:
                self.model = json.load(f)
        except e:
            raise ValueError("Model config not found! {}/{}.json".format(self.path, m_name))
        self.self_update()

    def self_update(self):
        conf = dict()
        conf.update(self.base)
        conf.update(self.model)
        self.conf = conf


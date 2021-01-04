import os
from lcutil.config_util import ConfigUtil as BaseConfig

class Config(BaseConfig):
    def __init__(self, d='data', f='config.ini'):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        super(Config, self).__init__(d, f)

    def get_nir_camera_config(self):
        section = 'nir_camera'
        return {
            'device': self.get_eval_config(section, 'device'),
        }
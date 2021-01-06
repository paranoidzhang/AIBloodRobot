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

    def get_thresh_params1_config(self):
        section = 'thresh_params1'
        return {
            'sidelength': self.get_eval_config(section, 'sidelength'),
            'threk': self.get_eval_config(section, 'threk'),
            'b_openside': self.get_eval_config(section, 'b_openside'),
            'b_closedside': self.get_eval_config(section, 'b_closedside'),
            'b_blurside': self.get_eval_config(section, 'b_blurside'),
            'gamma': self.get_eval_config(section, 'gamma'),
            'winsize': self.get_eval_config(section, 'winsize'),
            'blurside': self.get_eval_config(section, 'blurside'),
            'threfactor': self.get_eval_config(section, 'threfactor'),
            'erodeside': self.get_eval_config(section, 'erodeside'),
            'dilateside1': self.get_eval_config(section, 'dilateside1'),
            'dilateside2': self.get_eval_config(section, 'dilateside2'),
            'n_openside': self.get_eval_config(section, 'n_openside'),
            'n_closedside': self.get_eval_config(section, 'n_closedside'),
            'n_blurside': self.get_eval_config(section, 'n_blurside'),
        }

    def get_thresh_params2_config(self):
        section = 'thresh_params2'
        return {
            'sidelength': self.get_eval_config(section, 'sidelength'),
            'threk': self.get_eval_config(section, 'threk'),
            'b_openside': self.get_eval_config(section, 'b_openside'),
            'b_closedside': self.get_eval_config(section, 'b_closedside'),
            'b_blurside': self.get_eval_config(section, 'b_blurside'),
            'gamma': self.get_eval_config(section, 'gamma'),
            'winsize': self.get_eval_config(section, 'winsize'),
            'blurside': self.get_eval_config(section, 'blurside'),
            'threfactor': self.get_eval_config(section, 'threfactor'),
            'erodeside': self.get_eval_config(section, 'erodeside'),
            'dilateside1': self.get_eval_config(section, 'dilateside1'),
            'dilateside2': self.get_eval_config(section, 'dilateside2'),
            'n_openside': self.get_eval_config(section, 'n_openside'),
            'n_closedside': self.get_eval_config(section, 'n_closedside'),
            'n_blurside': self.get_eval_config(section, 'n_blurside'),
        }
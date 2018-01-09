"""Visual Utilities
Now only support visdom.
Problems:
TODO:
1. We want to add tensorboard support.
2. Now some functions like weight_ratio don't support Multiple GPU situation.
3. Add some more functions.
"""

import visdom
import torch
import numpy as np


from torch import nn

from torchnet.logger import VisdomLogger, VisdomPlotLogger

# Temporarily, don't consider multiple progress situation
# And in the current design, the Visual Class is designed for the oneshot situation

buffer = []

def list_sub_paras(layer):
    print("\n".join(layer.state_dict().keys()))

def list_sub_modules(layer):
    for l in layer.named_modules():
        print(l)

# =========hook def============
def debug(debugger):
    if debugger == 'pdb':
        import pdb
        pdb.set_trace()
    elif debugger == 'ipdb':
        import ipdb
        ipdb.set_trace()

def hook_pdb(*args):
    debug('pdb')


def hook_pre_forward(module, input):
    #only records input
    module.in_ = input


def hook_forward(module, input, output):
    #records input and output
    # if isinstance(module, nn.Linear):
    #     import pdb;pdb.set_trace()
    if not hasattr(module, 'out_'):
        module.out_ = output.data
    else:
        module.out_ = torch.cat([module.out_, output.data])
    # module.in_ = input

# def hook_backward_weight_ratio(module, grad):
#     Wv = module
#     module.ratio_Wv = torch.norm(grad)/
#     return grad
# ========register=========

def register_pre_forward_hook(module, hook):
    module.register_forward_pre_hook(hook)


def register_forward_hook(module, hook):
    module.register_forward_hook(hook)


def register_backward_hook(module, hook):
    module.register_backward_hook(hook)


class Visual(object):

    def __init__(self, model, env, port=8097):
        hook_grad_out = lambda module, grad_in, grad_out: grad_out

        self.model = model
        self.env = env
        # self.port = port
        self.viz = visdom.Visdom(env=env, port=port)
        self.hooks={
            # 'grad': hook_grad,
            'record_forward_in': hook_pre_forward,
            'record_forward': hook_forward,
        }

        self.regis_hook = {
            'forward_in': register_pre_forward_hook,
            'forward_out': register_forward_hook,
            'backward': register_backward_hook,
        }

        self.plots = []

    def get_layer(self, layer_index):
        # According to layer_index, get the layer of the model
        layers_index = layer_index.split('.')
        model = self.model
        ind_path = ''
        for ind in layers_index:
            # if ind.isdigit():
            #     ind = int(ind)
            try:
                model = model._modules[ind]
            except KeyError as e:
                print('\n',repr(e),'Current model is {},index should be in :'.format(ind_path))
                print('\t\n'.join(model.state_dict().keys()))
                import pdb; pdb.set_trace()
            finally:
                ind_path = ind_path + '.' + ind
        layer = model
        return layer

    def get_para(self, layer_index, para_name):
            layer = self.get_layer(layer_index)
            try:
                para = layer._parameters[para_name]

            except KeyError as e:
                print(repr(e))
                print('Hint:')
                print("\t layer: ")
                print("\t\tClass Info: ", layer.__class__)
                print("\t\tLayer Index: ", layer_index)
                print("\t para: ", list(layer._parameters.keys()))
                import pdb;pdb.set_trace()

            finally:
                return para

    def register_gradient_hook(self, layer_index, var):
        layer = self.get_layer(layer_index)
        if var in ['grad_in', 'grad_out']:
            hook = self.hooks[var]
            layer.register_backward_hook(hook)

        else:
            var = layer._parameters[var]
            var.register_hook(lambda grad:grad)

    def register_vis_featuremap(self, layer_index, where='record_forward'):
        """
        :param layer_index:
        :param where: record_forward(records in and out) / record_forward_in(only records in)
        """
        mapping = {
            'record_forward': 'forward_out',
            'record_forward_in': 'forward_in',
        }
        layer = self.get_layer(layer_index)
        hook = self.hooks[where]
        regis_idx = mapping[where]
        self.regis_hook[regis_idx](layer, hook)

    def vis_featuremap(self, layer_index, ch=None, wheres=('forward_out','forward_in')):
        mapping = {
            'forward_out': 'out_',
            'forward_in': 'in_',
        }

        wheres = tuple(wheres)
        layer = self.get_layer(layer_index)

        fmaps = []
        for where in wheres:
            fmap = getattr(layer, mapping[where])
            if isinstance(fmap, tuple):
                assert len(fmap)==1
                fmap = fmap[0]
            H, W = tuple(fmap.shape)[-2:]
            fmap = fmap.view(-1, H, W).cpu()
            fmap = np.asarray(fmap.data)
            fmap /= 2*np.amax(fmap)
            fmap += 0.5
            fmaps.append(fmap[0:1])
        # import pdb;pdb.set_trace()
        title = str(layer_index) + '_feature_map'
        # viz = VisdomLogger('images', env=self.env,opts={'title':title, 'caption':"\t".join(wheres)})
        self.viz.images(fmaps)
        # viz.log(fmaps,
        #         # padding=2,
        #         nrow=len(fmaps),
        #         # env=self.env,
        #         )

    def regis_weight_ratio_plot(self, layer_index, para_name, caption=''):
        """
        :param layer_index:
        :param para_name:
        :param epoch:
        """
        para = self.get_para(layer_index, para_name)
        logger_name = layer_index + '@' + para_name + '@grad_weight_ratio'
        self.plots.append(WeightRatio(para, self.env, logger_name, caption))

    def regis_norm_plot(self, layer_index, para_name, caption=''):
        """
        :param layer_index:
        :param para_name:
        :param epoch:
        """
        para = self.get_para(layer_index, para_name)
        logger_name = layer_index + '@' + para_name + '@norm'
        self.plots.append(Norm(para, self.env, logger_name, caption))

    def regis_mean_std(self, layer_index, caption=''):
        layer = self.get_layer(layer_index)
        logger_name = layer_index
        self.plots.append(MeanStd(layer, self.env, logger_name, caption))

    def update_plot(self):
        for ele in self.plots:
            ele.plot()

    def insert_pdb_layer(self, layer_index, debugger='pdb', where='forward_in'):
        """Insert (i)pdb into some layer (in/out)
        :param layer_index:
        :param debugger: ['pdb', 'ipdb']
        :param where: ['forward_in', 'forward_out']
        :return:
        """
        # Now don't support forward_out

        layer = self.get_layer(layer_index)
        try:
            self.regis_hook[where](layer, hook_pdb)
        except ValueError:
            raise ValueError("where arg in pdb_layer method should be pre or after.")


#How to cancel all hooks



# the content you want to record is abstractd into a class
class WeightRatio(object):
    def __init__(self, para, env, logger_name, caption=''):
        # import pdb;pdb.set_trace()
        self.para = para
        self.para.register_hook(lambda grad: grad)
        self.logger = VisdomPlotLogger('line', env=env, opts={'title': caption + '\n' + logger_name , 'caption': caption})
        self.iter_n = 0

    def plot(self):
        ratio = torch.norm(self.para.grad.data, 2) / torch.norm(self.para.data, 2)
        self.logger.log(self.iter_n, ratio)
        self.iter_n += 1

class MeanStd(object):
    """ plot the mean and std of the layer output """
    def __init__(self, layer, env, logger_name, caption):
        self.layer = layer
        self.layer.register_forward_hook(hook_forward)
        self.mean_logger = VisdomPlotLogger('line', env=env, opts={'title': 'mean_' + caption+ logger_name, 'caption': caption})
        self.std_logger = VisdomPlotLogger('line', env=env, opts={'title': 'std_' + caption + logger_name, 'caption': caption})
        self.iter_n = 0

    def plot(self):

        mean = torch.mean(self.layer.out_)
        std = torch.std(self.layer.out_)
        del self.layer.out_
        self.mean_logger.log(self.iter_n, mean)
        self.std_logger.log(self.iter_n, std)
        self.iter_n += 1

class Norm(object):

    def __init__(self, para, env, logger_name, caption=''):
        self.para = para
        self.logger = VisdomPlotLogger('line', env=env, opts={'title': 'norm_' + caption + logger_name, 'caption': caption})
        self.iter_n = 0

    def plot(self):
        self.logger.log(self.iter_n, torch.mean(self.para.data**2))
        self.iter_n += 1


# Manage viz
class VisualManager(object):

    def __init__(self):
        self.viz_dict = {}

    def create_viz(self, name, model, env, port):
        viz = Visual(model, env, port)
        assert name not in self.viz_dict, "Visual Manager: the name in viz_dict already exist"
        self.viz_dict[name] = viz


def get_viz(name=None):
    name = name if name else 'root'
    return viz_manager.viz_dict[name]


def create_viz(name, model, env='debug', port=8097):
    name = name if name else 'root'
    viz_manager.create_viz(name, model, env, port)
    return get_viz(name)

viz_manager = VisualManager()
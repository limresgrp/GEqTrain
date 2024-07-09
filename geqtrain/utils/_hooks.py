import torch
import torch.nn as nn
from functools import partial
import warnings
from typing import Callable, Optional, Union, Tuple, List, Dict
import numpy as np
import wandb





##################
### IN TESTING ###
##################

################################################

# def register_hooks(self):
#     list_of_hookables = get_all_modules(self.model)
#     hooks = []
#     for i, (name, layer) in enumerate(list_of_hookables):
#         try:
#             # shapes_printer = partial(print_shape, i, [])) # 'LayerNorm'
#             # hooks.append(layer.register_forward_hook(shapes_printer))

#             if self.is_wandb_trainer:
#                 # f = partial(gather_stats, i, ["ShiftedSoftPlusModule"] , ['LayerNorm'])
#                 # hooks.append(layer.register_forward_hook(f))
#                 # if model has 2d weight:
#                 if layer.weight.dim() >= 2:
#                     f = partial(log_gradients, i)
#                     hooks.append(layer.register_full_backward_hook(f))


#         except:
#             print(f"Cant hook on {name}")
#             pass

#     self.hookss = hooks

# def deregister_hooks(self):
#     for hook in self.hookss:
#         hook.remove()


# def gather_stats(i, filter_in, filter_out, mod, inp, out):
#     '''
#     filter: type of layer to skip

#     '''
#     if mod.__class__.__name__ in filter_out:
#         return

#     # try:
#     name = f"{i}_{mod.__class__.__name__}"
#     if str(mod.__class__.__name__) in filter_in:
#         acts = out.detach()
#         # wandb.log({f"{name}_mean": acts.mean()})
#         # wandb.log({f"{name}_std": acts.std()})
#         # wandb.log({f"{name}_%_dead": acts.abs().histc(40,0,10)})   # adds hist of abs vals of activations, 50 bins
#         hist = np.histogram(acts.to('cpu'))
#         wandb.log({f"{name}_dist": wandb.Histogram(np_histogram=hist)})

#     # except:
#     #     pass


#####################################################


def get_all_modules(model):
    modules = []

    def has_children(module):
        return len(list(module.named_children()))

    def recurse(mod, prefix=''):
        for name, child in mod.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Module):
                if has_children(child):
                    recurse(child, full_name)
                else:
                    modules.append((full_name, child))

    recurse(model)
    return modules


def get_all_hookable_modules(model):

    '''
    The id of the module is defined by the order of where the module
    is defined in the ctor
    '''

    modules = get_all_modules(model)
    hookable_modules = []
    f = lambda x : None
    for id, (name, layer) in enumerate(modules):
        try:
            h = layer.register_forward_hook(f)
            h.remove()
            hookable_modules.append((id, name, layer))
        except:
            warnings.warn(f'Cannot attach hook to layer with ID {id}: {name}')
    return hookable_modules


def print_stats(hook_handler, module_id, module_name, module, input, output):

    def print_tensor(t):
        return f"shape: {t.shape}, mean: {t.mean().item()}, std: {t.std().item()}, min: {t.min().item()}, max {t.max().item()}"

    if isinstance(input, tuple): # always hit
        input = input[0]
    if isinstance(input, dict):
        input = input[module.field] # grab input field
    if isinstance(output, dict):
        output = output[module.out_field] # grab output field

    input  = input.detach()
    output = output.detach()

    print(f"ModuleID: {module_id} of type {module} in {module_name}")
    print("Input: ", print_tensor(input))
    print("Output ", print_tensor(output), "\n")

    if not hasattr(hook_handler, "stats"):
        hook_handler.stats = {} # stores means and stds of hookable layers

    key = f"{module_id}_{module_name}"

    # wandb.log({f"{module_id}_{module_name}":
    #          wandb.Histogram(output.tolist())})

    if key not in hook_handler.stats:
        hook_handler.stats[key] = {'output_mean': [], 'output_std': []}

    hook_handler.stats[key]['output_mean'].append(output.mean().item())
    hook_handler.stats[key]['output_std'].append(output.std().item())

    columns = ['output_mean', 'output_std']
    n = len(hook_handler.stats[key]['output_mean'])
    xs = list(range(n))
    ys = [hook_handler.stats[key]['output_mean'], hook_handler.stats[key]['output_std']]

    wandb.log(
        {f"{key}" :wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=columns,
            title=f"Activations mean/std {key}")
        }
    )


'''
To create a new hook function you must register it into function_mappings dict as:
- k: name of the method, this key is parsed as string from the yaml
- value, the actual python method
'''
available_hooks = {
    "print_stats" : print_stats,
}


class ForwardHookHandler:
    '''

    pytorch forward hook handler

    hooks a hook_func to a set of modules of model
    if filter_in is empty all hookable modules are hooked
    if filter_out is not empty the modules inside it are not hooked
    hook_func is executed when the module.forward() method is called

    hook_func interface must be:
        - hook_handler: ptr to its ForwardHookHandler
        - module_id: index of module when enumerating hookable_layers
        - module_name: str name of the module
        - module: the pytorch module itself
        - input: the input of the hooked module; input is always a tuple with: (input,)
        - output: the output of the hooked module
    '''
    def __init__(self, trainer, callables: List[Dict] = []):
        model = trainer.model
        self.hooks = []
        self.hookable_layers: List = get_all_hookable_modules(model)
        self.clean_module_name = lambda s : s.split('(')[0].lower()

        for hook_data in callables:
            callable_name, filter_in, filter_out = hook_data["callable_name"], hook_data["filter_in"], hook_data["filter_out"]
            filter_in = filter_in or []
            filter_out = filter_out or []
            callable_ = available_hooks.get(str(callable_name), None)
            if callable_:
                self._register_hook(
                    hook_func  =  callable_,
                    filter_in  = filter_in,
                    filter_out = filter_out,
                )

    def _register_hook(self, hook_func: callable, filter_in: List = [], filter_out: List = []):
        filter_in  = [self.clean_module_name(s) for s in filter_in]
        filter_out = [self.clean_module_name(s) for s in filter_out]

        for (id, name, layer) in self.hookable_layers:
            is_module_in_filter_in = self.clean_module_name(str(layer)) in filter_in
            is_module_in_filter_out = self.clean_module_name(str(layer)) in filter_out
            if is_module_in_filter_out:
                continue
            elif (not filter_in) or is_module_in_filter_in:
                callable = partial(hook_func, self, id, name)
                self.hooks.append(layer.register_forward_hook(callable))
            else:
                raise ValueError(f'IDK what to do with {self.clean_module_name(str(layer))} (aka {str(layer)})')

    def deregister_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.deregister_hooks()


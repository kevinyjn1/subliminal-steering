import torch
import torch.nn as nn

class SteeredModelWrapper(nn.Module):
    def __init__(self, base_model, steering_vector, alpha=1.0, target_layer=-1):
        super().__init__()
        self.base_model = base_model
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.target_layer = target_layer
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        layer = self.base_model.model.layers[self.target_layer]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            return hidden_states + self.alpha * self.steering_vector.to(hidden_states.device)

        self.hook_handle = layer.mlp.register_forward_hook(hook_fn)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def __del__(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

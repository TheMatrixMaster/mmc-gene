import collections

def move_batch_input_to_device(x, device):
    for key, item in x.items():
        if isinstance(item, collections.abc.Mapping):
            x[key] = move_batch_input_to_device(x.get(key, {}), device)
        else:
            x[key] = item.to(device)
    return x

def get_output_activation_from_loss(loss_name):
    if loss_name == "cross_entropy":
        return "softmax"
    elif loss_name == "mtl_bceloss":
        return "sigmoid"
    else:
        raise ValueError("Loss {} not implemented".format(loss_name))
    
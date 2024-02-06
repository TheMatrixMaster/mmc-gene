import collections

def move_batch_input_to_device(x, device):
    for key, item in x.items():
        if isinstance(item, collections.abc.Mapping):
            x[key] = move_batch_input_to_device(x.get(key, {}), device)
        else:
            x[key] = item.to(device)
    return x

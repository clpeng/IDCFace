
def setLayerHock(net, layers_name, features_in_hook, features_out_hook):
    def hook(module, fea_in, fea_out):
        # features_in_hook.append(fea_in[0])
        # print('len_feature_out', len(features_out_hook))
        features_out_hook.append(fea_out)
        return None
    for name, module in net.named_modules():
        for layer_name in layers_name:
            if name == layer_name:
                module.register_forward_hook(hook=hook)



def setLayerHockArray(net, layers_name, features_in_hook, features_out_hook,phase_key, rand_idx=0):
    def get_activation(name):
        def hook(model, input, output):
            features_out_hook[phase_key + '-' + name] = output[rand_idx].detach()

        return hook
    for name, module in net.named_modules():
        for layer_name in layers_name:
            if name == layer_name:
                module.register_forward_hook(hook=get_activation(name))


def setLayerHock_item(net, layers_name):
    feature_out = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_out[name] = output[0].detach()

        return hook
    for name, module in net.named_modules():
        for layer_name in layers_name:
            if name == layer_name:
                module.register_forward_hook(get_activation(layer_name))

    return feature_out

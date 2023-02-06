import torch
from torch import nn
from functools import partial
from typing import Union


# Select Accelerator to train and infer models
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
def get_out_channels(layers: list):
    get_out = lambda obj, name: getattr(obj, name) if hasattr(obj, name) else False

    # Go bottom-top until find a layer with the output channels number
    for layer in reversed(layers):
        out = get_out(layer, "out_channels") or get_out(layer, "num_features")
        if out: break
    return out


class ResBlock(nn.Module):
    def __init__(self, layers: list):
        """
        Class that implements a Resnet block like those showed on the article.

        Normally the shortcut has a linear transformation layer to down sample
        the input to match the output.

        Although in the article nothing is showed, so I assumed the shortcut is directly connect to the output,
        but implementing that makes the program slower due the in-depth zero tensors needed.
        So, to down sample the input was used a Convolution to down sample and learn through the shortcut, is basically
        the linear transformation is a non-linear because was used a ReLU actiavtion function.

        To match the filters size was done the difference and zero padded the input or the output. In the
        end input and output are added.

        :param layers: List of Keras layers
        :param args: List of extra arguments
        :param kwargs: Dictionary of extra arguments
        """
        super(ResBlock, self).__init__()
        self.layers = layers

    def forward(self, x):
        net = x.clone()

        for layer in self.layers:
            net = layer(net)

        # Calculating the kernel size needed to downsample
        out_channels = get_out_channels(self.layers)
        k_size = (x.shape[-2] - net.shape[-2] + 1, x.shape[-1] - net.shape[-1] + 1)

        # Apply an convolution using its identity function (delta function)
        conv = nn.Conv2d(x.shape[1], out_channels, k_size).to(DEVICE)
        nn.init.dirac_(conv.weight)

        return net + conv(x)

class StridedConv2D(nn.LazyConv2d):
    def __init__(self, *args, **kwargs):
        """
        Class that is a custom type of 2D Convolutions, then could be distiguished to apply operations
        :param args: List of extra parameters
        :param kwargs: Dicitionary of extra parameters
        """
        super(StridedConv2D, self).__init__(*args, **kwargs)


def resnet_block1():
    """
    Function that constructs the b1 block showed on article
    :return: Resblock object initialized
    """
    return ResBlock(
        nn.ModuleList([
            nn.LazyBatchNorm2d(device=DEVICE),
            StridedConv2D(16, (3, 3), stride=1, device=DEVICE),
            nn.ReLU(),
            nn.LazyBatchNorm2d(device=DEVICE),
            nn.LazyConv2d(16, (3, 3), stride=1,device=DEVICE),
            nn.ReLU(),
        ])
    ).to(DEVICE)


def resnet_block2():
    """
        Function that constructs the b2 block showed on article
        :return: Resblock object initialized
        """
    return ResBlock(
        nn.ModuleList([
            nn.LazyBatchNorm2d(device=DEVICE),
            StridedConv2D(
                16,
                (3, 3),
                stride=1,
                device=DEVICE
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(device=DEVICE),
            nn.LazyConv2d(16, (3, 3), stride=1,device=DEVICE),
            nn.ReLU(),
            nn.LazyBatchNorm2d(device=DEVICE),
            nn.LazyConv2d(16, (3, 3), stride=1,device=DEVICE),
            nn.ReLU(),
        ])
    ).to(DEVICE)


def resnet_block3():
    """
        Function that constructs the b3 block showed on article
        :return: Resblock object initialized
        """
    return ResBlock(
        nn.ModuleList([
            nn.LazyBatchNorm2d(device=DEVICE),
            StridedConv2D(16, (3, 3), stride=1,device=DEVICE),
            nn.ReLU(),
            nn.LazyBatchNorm2d(device=DEVICE),
            nn.LazyConv2d(16, (1, 1), stride=1,device=DEVICE),
            nn.ReLU(),
        ])
    ).to(DEVICE)


def resnet_block4():
    """
        Function that constructs the b4 block showed on article
        :return: Resblock object initialized
        """
    return ResBlock(
        nn.ModuleList([
            nn.LazyBatchNorm2d(device=DEVICE),
            StridedConv2D(
                16,
                (3, 3),
                stride=1,
                device=DEVICE
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(device=DEVICE),
            nn.LazyConv2d(16, (1, 1), stride=1, device=DEVICE),
            nn.ReLU(),
            nn.LazyBatchNorm2d(device=DEVICE),
            nn.LazyConv2d(16, (3, 3), stride=1, device=DEVICE),
            nn.ReLU(),
        ])
    ).to(DEVICE)


def concat(t1: Union[list, ResBlock], t2: Union[list, ResBlock]):
    """
    Function that makes the concatenation (non-terminal node) of 2 model blocks (resnet blocks)

    :param t1: Object Resblock (terminal)
    :param t2: Another object Resblock (terminal)
    :return: List with Resblock objects inside
    """
    t_lists = list()

    for model in [t1, t2]:
        if isinstance(model, list):
            for k in model:
                t_lists.append(k)
        else:
            t_lists.append(model)

    return t_lists


def stride_factor(t):
    """
    Function that doubles the stride (non-terminal node) of a StrideConv2D
    :param t: ResBlock object (terminal)
    :return: List of Resblock objects
    """
    return apply_ops(t, partial(apply_stride, 2), "stride", (StridedConv2D,))


def two_times_filters(node):
    """
    Function that doubles the filter size (non-terminal node) of Convolutions
    :param node: Any non-terminal node
    :return: List of Resblock objects
    """
    return x_times_filters(node, 2)


def three_times_filters(node):
    """
    Function that triples the filter size (non-terminal node) of Convolutions
    :param node: Any non-terminal node
    :return: List of Resblock objects
    """
    return x_times_filters(node, 3)


def x_times_filters(node, x: int):
    """
    Function that X times the filter size (non-terminal node) of Convolutions

    If the node is another x_times_filters non-terminal node, then they sum up (as explained on the article)

    If the node is a list means some Resblocks are concatenated, then apply this primitive to the left most
    Resblock.

    :param x: Number of X times
    :param node: Any non-terminal node
    :return: List of Resblock objects
    """
    types = (nn.LazyConv2d, StridedConv2D)

    if isinstance(node, int):
        return node + x

    elif isinstance(node, list):
        t = apply_ops(node[0], partial(apply_filters, x), "out_channels", types)
        return [t] + node[1:]

    return apply_ops(node, partial(apply_filters, x), "out_channels", types)


def apply_ops(model, op_func: partial, attr: str, types: tuple):
    """
    Function that applies an operation:
     - apply_filters in case of x_times_filtesr
     - apply_stride in case of stride_factor

    To apply those functions were used partial functions to specify, without running, the factor of strides or
    filtes to be multiplied.

    In this fucntion those partial functions are executed, passing the current value
    (since the factor was passed previoulys on other functions) and applying the operations.

    :param model: Node to apply the factor
    :param op_func: Function that calculates the new attribute value based on the factor
    :param attr: Attribute which value will change
    :param types: Valid layer types to be applied the operation
    :return: Modified node
    """
    for layer in model.layers:
        if isinstance(layer, types):
            val = getattr(layer, attr)

            # Execute partial function to adjust current value of layer attr
            new_val = op_func(val)

            setattr(layer, attr, new_val)
    return model


def apply_filters(factor, cur_value, default_val=16):
    """
    Function that applied factor to the current value.

    Although, a default_val is provided to be possible to calculate the right new value.

    Imagine we have a convolution layer with filters = 16 and our factor is 2+3 = 5 because
    one 2 times and other 3 times added up.

    The problem is that the tree is processed recursively so if we multiply directly the factor with
    the current value would be something like: 16 * 2 * 3 = 96

    The result is wrong because we want: 16 * (2+3) = 80

    To circumvent that we always multiply the default value by the calculated factor and if
    the current value is not the default one (means that filters were already multiplied) just add
    the current value.

    In our example this would be: 16 * 2 = 32 then 16 * 3 + 32 = 80

    The default value for the filters is 16 because was said on the article.

    :param factor: Factor to be applied
    :param cur_value: Current value to multiplied by the factor
    :param default_val: Default value of filters number
    :return: New filters number value
    """
    new_val = default_val * factor

    if cur_value != default_val:
        new_val += cur_value

    return new_val


def apply_stride(factor, cur_value):
    """
    Function that applies the double to the current of strides in a StrideConv2D

    Isn't needed to accumulate like the apply_filters() function because is always doubling the current value

    :param factor: Normally value 2 to double the current value
    :param cur_value: Current value
    :return: New value for the stride attribute
    """
    # If strides is a list or tuple needs to be iterated to scale the strides value
    if isinstance(cur_value, (list, tuple)):
        new_val = [i * factor for i in cur_value]
    else:
        new_val = cur_value * factor

    return new_val

# flake8: noqa
import torch
import numpy as np
import time
import warnings
from .apputils.handlers import handlers
from .apputils.trace import trace
import logging
import typing
from collections import defaultdict
import torch.nn as nn

from .apputils.jit_handles import (
    addmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    get_jit_model_analysis,
    matmul_flop_jit,
)

# A dictionary that maps supported operations to their flop count jit handles.
_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::addmm": addmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
}


cuda = False
device = torch.device('cuda' if cuda else 'cpu')


__all__ = ['profile_macs', 'get_inf_time', 'profile_flops']


def profile_macs(model, args=(), kwargs=None, reduction=sum):
    results = dict()

    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node] = func(node)
                break
        else:
            warnings.warn('No handlers found: "{}". Skipped.'.format(
                node.operator))

    if reduction is not None:
        return reduction(results.values())
    else:
        return results



def get_inf_time(model, input_size):
    # reference: https://medium.com/@auro_227/timing-your-pytorch-code-fragments-e1a556e81f2
    if len(input_size) == 3:
        input_size = (1, input_size[0], input_size[1], input_size[2])

    # the input size must be 4
    assert len(input_size) == 4

    im_input = np.random.random(input_size)
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        # set GPU mode
        start = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
        end = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
        model = model.to(device)
        im_input = im_input.to(device)

        with torch.no_grad():
            start.record()
            model(im_input)
            end.record()
            # need to set synchronize way during profiling
            torch.cuda.synchronize()
            tic = start.elapsed_time(end)
    else:
        # set CPU mode with no CUDA enabled
        model = model.to(device)
        im_input = im_input.to(device)
        start_time = time.time()
        model(im_input)
        tic = time.time() - start_time

    return tic

def profile_flops(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    supported_ops: typing.Union[typing.Dict[str, typing.Callable], None] = None,
) -> typing.Tuple[typing.DefaultDict[str, float], typing.Counter[str]]:
    """
    Given a model and an input to the model, compute the Gflops of the given
    model. Note the input should have a batch size of 1.

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : By default, we count flops
            for convolution layers, fully connected layers, torch.matmul and
            torch.einsum operations. We define a FLOP as a single atomic
            Multiply-Add. Users can provide customized supported_ops for
            counting flops if desired.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            skipped operations.
    # from https://github.com/facebookresearch/fvcore/blob/b54a5ed9ce725a5a389ec6eaeead131daacd2ec8/fvcore/nn/flop_count.py
    """
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."
    if not supported_ops:
        supported_ops = _SUPPORTED_OPS.copy()

    # Run flop count.
    total_flop_counter, skipped_ops = get_jit_model_analysis(
        model, inputs, supported_ops
    )

    # Log for skipped operations.
    if len(skipped_ops) > 0:
        for op, freq in skipped_ops.items():
            logging.warning("Skipped operation {} {} time(s)".format(op, freq))

    # Convert flop count to gigaflops.
    final_count = defaultdict(float)
    for op in total_flop_counter:
        final_count[op] = total_flop_counter[op] / 1e9

    return final_count, skipped_ops

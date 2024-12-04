import torch
from torch._higher_order_ops.prim_hop_base import PrimHOPBase


supported_ops = {torch.add}


def _validate_args(operands):
    pass


class ForeachMap(PrimHOPBase):
    def __init__(self):
        return super().__init__("foreach_map")

    def __call__(self, fn, operands, *unused, **kwargs):
        # Manually create a fn with the same semantics as a foreach_* op
        _validate_args(operands)
        # fn = FunctionWithNoFreeVars(foreach_map_fn)
        return super().__call__(fn, operands, **kwargs)


_foreach_map = ForeachMap()


def foreach_map(op, operands, *unused, **kwargs):
    from torch._dynamo.polyfills import foreach_map_fn

    args = (op,) + operands
    return _foreach_map(foreach_map_fn, args, kwargs)

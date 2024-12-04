# mypy: allow-untyped-defs
import copy
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from ._compatibility import compatibility
from .node import Node


__all__ = [
    "preserve_node_meta",
    "has_preserved_node_meta",
    "set_stack_trace",
    "set_grad_fn_seq_nr",
    "reset_grad_fn_seq_nr",
    "format_stack",
    "set_current_meta",
    "get_current_meta",
]

current_meta: Dict[str, Any] = {}
should_preserve_node_meta = False


class NodeSource:
    def __init__(self, node: Optional[Node], pass_name: str = "", action: str = ""):
        self.pass_name = pass_name
        self.action = action

        if action not in ["create", "replace", ""]:
            warnings.warn(f"unknown action for node source, {action}")

        if node:
            self.node_name = node.name
            self.target = str(node.target)
            self.graph_id = id(node.graph)
            self.from_node: List[NodeSource] = (
                copy.deepcopy(node.meta["from_node"])
                if "from_node" in node.meta
                else []
            )
        else:
            self.node_name = ""
            self.target = ""
            self.graph_id = -1
            self.from_node: List[NodeSource] = []

    def __repr__(self):
        return self.print_readable()

    def print_readable(self, indent=0):
        if indent > 36:
            return ""
        result = ""
        result += (
            " " * indent
            + f"(node_name={self.node_name}, pass_name={self.pass_name}, action={self.action}, graph_id={self.graph_id})\n"
        )
        for item in self.from_node:
            result += item.print_readable(indent + 4)
        return result

    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        return {
            "node_name": self.node_name,
            "target": self.target,
            "graph_id": self.graph_id,
            "pass_name": self.pass_name,
            "action": self.action,
            "from_node": [node.to_dict() for node in self.from_node],
        }


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta():
    global should_preserve_node_meta
    global current_meta

    saved_should_preserve_node_meta = should_preserve_node_meta
    # Shallow copy is OK since fields of current_meta are not mutated
    saved_current_meta = current_meta.copy()
    try:
        should_preserve_node_meta = True
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta
        current_meta = saved_current_meta


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: List[str]):
    global current_meta

    if should_preserve_node_meta and stack:
        current_meta["stack_trace"] = "".join(stack)


@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr):
    global current_meta

    if should_preserve_node_meta:
        # The seq_nr is captured by eager mode in the grad_fn during forward
        current_meta["grad_fn_seq_nr"] = current_meta.get("grad_fn_seq_nr", []) + [
            seq_nr
        ]
        current_meta["in_grad_fn"] = current_meta.get("in_grad_fn", 0) + 1


@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr():
    # NB: reset state properly, this would be helpful towards supporting
    #     reentrant autograd if we actually wanted to do that.
    global current_meta
    if should_preserve_node_meta:
        current_level = current_meta.get("in_grad_fn", 0)
        assert current_level > 0
        if current_level == 1:
            del current_meta["in_grad_fn"]
            del current_meta["grad_fn_seq_nr"]
        else:
            current_meta["in_grad_fn"] = current_level - 1
            current_meta["grad_fn_seq_nr"] = current_meta["grad_fn_seq_nr"][:-1]


@compatibility(is_backward_compatible=False)
def format_stack() -> List[str]:
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node, pass_name=""):
    global current_meta
    if should_preserve_node_meta and node.meta:
        saved_meta = current_meta
        try:
            current_meta = node.meta.copy()

            # Append NodeSource onto "from_node" for provenance tracking
            if "from_node" not in current_meta:
                current_meta["from_node"] = [NodeSource(node, pass_name, "create")]
            elif current_meta["from_node"][-1].node_name != node.name:
                current_meta["from_node"] = current_meta["from_node"] + [
                    NodeSource(node, pass_name, "create")
                ]

            yield
        finally:
            current_meta = saved_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> Dict[str, Any]:
    return current_meta

# Owner(s): ["module: fx"]
import torch
from torch.fx.traceback import NodeSource
from torch.testing._internal.common_utils import TestCase


class TestFXNodeSource(TestCase):
    def test_node_source(self):
        node_source = NodeSource(node=None, pass_name="test_pass", action="create")
        self.assertExpectedInline(
            node_source.print_readable().strip(),
            """(node_name=, pass_name=test_pass, action=create, graph_id=-1)""",
        )
        dummy_source_dict = {
            "node_name": "",
            "target": "",
            "pass_name": "test_pass",
            "action": "create",
            "graph_id": -1,
            "from_node": [],
        }
        self.assertEqual(
            node_source.to_dict(),
            dummy_source_dict,
        )

        # Dummy node
        node = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="add",
            op="call_function",
            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
            args=(torch.tensor(3), torch.tensor(4)),
            kwargs={},
        )
        node.meta["from_node"] = [node_source]

        graph_id = id(node.graph)
        node_source = NodeSource(node=node, pass_name="test_pass", action="create")
        self.assertExpectedInline(
            node_source.print_readable().strip(),
            f"""\
(node_name=add, pass_name=test_pass, action=create, graph_id={graph_id})
    (node_name=, pass_name=test_pass, action=create, graph_id=-1)""",
        )
        self.assertEqual(
            node_source.to_dict(),
            {
                "node_name": "add",
                "target": "aten.add.Tensor",
                "pass_name": "test_pass",
                "action": "create",
                "graph_id": graph_id,
                "from_node": [dummy_source_dict],
            },
        )

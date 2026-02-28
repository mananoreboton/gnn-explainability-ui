"""GNNExplainer wrappers using torch_geometric.explain API (PyG 2.3+)."""

import torch
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer as GNNExplainerAlgo


def _create_explainer(model, task_level, epochs=200, log=False, **kwargs):
    """Create Explainer with GNNExplainer algorithm for the given task level."""
    algorithm = GNNExplainerAlgo(epochs=epochs, log=log, **kwargs)
    return Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="phenomenon",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level=task_level,
            return_type="log_probs",
        ),
    )


class TargetedGNNExplainer:
    """Wrapper that provides explain_node_with_target using PyG Explainer API."""

    def __init__(self, model, epochs=200, log=False, **kwargs):
        self.model = model
        self.epochs = epochs
        self.log = log
        self.coeffs = {}
        self.coeffs.update(kwargs)

    def explain_node_with_target(self, node_idx, x, edge_index, target_class, **kwargs):
        explainer = _create_explainer(
            self.model,
            task_level="node",
            epochs=self.epochs,
            log=self.log,
            **self.coeffs,
        )
        target = torch.tensor([target_class], dtype=torch.long, device=x.device)
        explanation = explainer(x, edge_index, target=target, index=node_idx, **kwargs)
        node_feat_mask = explanation.node_feat_mask
        edge_mask = explanation.edge_mask
        if node_feat_mask is None:
            node_feat_mask = torch.ones(1, x.size(1), device=x.device)
        if edge_mask is None:
            edge_mask = torch.ones(edge_index.size(1), device=x.device)
        if node_feat_mask.dim() == 1:
            node_feat_mask = node_feat_mask.unsqueeze(0)
        return node_feat_mask, edge_mask


class TargetedGNNExplainerGraph:
    """Wrapper that provides explain_with_target using PyG Explainer API."""

    def __init__(self, model, epochs=200, log=False, **kwargs):
        self.model = model
        self.epochs = epochs
        self.log = log
        self.coeffs = {}
        self.coeffs.update(kwargs)

    def explain_with_target(self, x, edge_index, target_class, **kwargs):
        explainer = _create_explainer(
            self.model,
            task_level="graph",
            epochs=self.epochs,
            log=self.log,
            **self.coeffs,
        )
        target = torch.tensor([target_class], dtype=torch.long, device=x.device)
        explanation = explainer(x, edge_index, target=target, **kwargs)
        node_feat_mask = explanation.node_feat_mask
        edge_mask = explanation.edge_mask
        if node_feat_mask is None:
            node_feat_mask = torch.ones(1, x.size(1), device=x.device)
        if edge_mask is None:
            edge_mask = torch.ones(edge_index.size(1), device=x.device)
        if node_feat_mask.dim() == 1:
            node_feat_mask = node_feat_mask.unsqueeze(0)
        return node_feat_mask, edge_mask

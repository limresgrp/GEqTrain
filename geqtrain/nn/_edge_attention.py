import torch


def edge_group_self_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    edge_center: torch.Tensor,
    num_nodes: int,
    inv_sqrtd: float,
    logit_clip: float = 0.0,
) -> torch.Tensor:
    """Self-attention weights for all edge pairs that share the same center node.

    Returns one scalar weight per edge and attention head. The attention is
    query-key self-attention within each center-node edge group, then summed
    over query edges so the following edge-to-node reduction remains a weighted
    sum over value edges.
    """
    num_edges = queries.shape[0]
    num_heads = queries.shape[1]
    weights = queries.new_zeros((num_edges, num_heads))
    for center in range(num_nodes):
        edge_mask = edge_center == center
        if not torch.any(edge_mask):
            continue
        q = queries[edge_mask]
        k = keys[edge_mask]
        logits = torch.einsum("ihd,jhd->ihj", q, k) * inv_sqrtd
        if logit_clip > 0.0:
            logits = torch.clamp(logits, min=-logit_clip, max=logit_clip)
        attn = torch.softmax(logits, dim=-1)
        weights[edge_mask] = attn.sum(dim=0).transpose(0, 1)
    return weights

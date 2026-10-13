"""
Graph utilities for the GAT-based NFO encoder.

The rest of the NFO pipeline (tokenizer, dataset, FeatureGenerator) speaks
post-order token sequences padded to a fixed length. To plug a GNN encoder
into that pipeline without churning every call site, we reconstruct the
feature tree from each row of the token tensor at the encoder's forward
pass using a per-vocab arity table, then batch the resulting trees into a
PyTorch Geometric ``Batch``.

Edge convention: each parent-child relation becomes two directed edges
(parent -> child and child -> parent) so message passing flows both ways.
Self loops are added by GATConv itself (add_self_loops=True is the PyG
default), so we do not add them here.
"""

from typing import List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, Batch


def build_arity_table(tokenizer, op_arity) -> np.ndarray:
    """Map token id -> arity (0 for leaves / PAD / start).

    tokenizer:   a Tokenizer with the LabelEncoder that produced the ids.
    op_arity:    dict-like mapping op_name -> arity (env.op_arity).

    Returns an int64 numpy array of length tokenizer.vocab_size + 1
    (the +1 covers tokenizer.start_idx, which sits at the end).
    """
    size = tokenizer.vocab_size + 1
    table = np.zeros(size, dtype=np.int64)
    classes = tokenizer._encoder.classes_  # ordered token strings
    for token, idx in zip(classes, tokenizer._encoder.transform(classes)):
        table[int(idx)] = int(op_arity.get(token, 0))
    # start_idx is the sentinel id at position vocab_size; arity 0
    return table


def _seq_to_tree_edges(token_ids: np.ndarray, arity_table: np.ndarray, padding_idx: int):
    """Reconstruct parent->child edges from a single post-order sequence.

    Returns:
        node_token_ids:  list[int]   tokens that participate as graph nodes
        edges:           list[(parent_local_idx, child_local_idx)]
    The local indices are positions within the per-graph node list.

    If the sequence is malformed (stack does not collapse to a single root),
    we fall back to a chain: every non-pad token in order, parent at the end.
    A single isolated node is also valid (e.g. a bare feature like ``"a"``).
    """
    nodes: List[int] = []
    edges: List[tuple] = []
    stack: List[int] = []  # holds local node indices

    for tid in token_ids:
        tid = int(tid)
        if tid == padding_idx:
            break
        arity = int(arity_table[tid])
        local_idx = len(nodes)
        nodes.append(tid)
        if arity == 0:
            stack.append(local_idx)
            continue
        if len(stack) < arity:
            # malformed: not enough children on the stack. Treat this token
            # as a leaf so we still produce a usable graph.
            stack.append(local_idx)
            continue
        children = stack[-arity:]
        stack = stack[:-arity]
        for child_idx in children:
            edges.append((local_idx, child_idx))
        stack.append(local_idx)

    if not nodes:
        # empty / fully padded sequence: emit a single PAD node so PyG does
        # not see an empty graph (which breaks batching).
        nodes = [int(padding_idx)]

    return nodes, edges


def seqs_to_pyg_batch(
    seqs: torch.Tensor,
    arity_table: np.ndarray,
    padding_idx: int,
    device: torch.device,
) -> Batch:
    """Convert a (B, T) long token tensor to a PyG Batch of B graphs.

    Each ``Data`` carries:
        x:           (num_nodes,) long token ids
        edge_index:  (2, num_edges) long, bidirectional parent<->child
    """
    seqs_np = seqs.detach().cpu().numpy()
    data_list: List[Data] = []
    for row in seqs_np:
        nodes, edges = _seq_to_tree_edges(row, arity_table, padding_idx)
        x = torch.as_tensor(nodes, dtype=torch.long)
        if edges:
            ei = torch.as_tensor(edges, dtype=torch.long).t().contiguous()
            # bidirectional: stack (parent->child) and (child->parent)
            ei_rev = ei.flip(0)
            edge_index = torch.cat([ei, ei_rev], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    batch = Batch.from_data_list(data_list).to(device)
    return batch

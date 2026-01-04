import copy
from typing import List, Optional

import numpy as np
import torch

_SKIP_SUFFIX = {"num_batches_tracked", "running_mean", "running_var"}


def _is_param_key(key: str, tensor: torch.Tensor) -> bool:
    """Trainable parameter key (skip BN running stats / counters)."""
    return tensor.dtype in (torch.float16, torch.float32, torch.float64) and key.split(".")[-1] not in _SKIP_SUFFIX


def _flatten_state_dict(state_dict: dict, template_state_dict: dict, device: torch.device) -> torch.Tensor:
    """
    Flatten a model (state_dict) into 1D tensor following template_state_dict order.

    Only trainable float parameters are included, to match Scope's original vectorize_net(net.parameters()).
    """
    flat = []
    for k, tmpl in template_state_dict.items():
        if not _is_param_key(k, tmpl):
            continue
        v = state_dict[k].to(device)
        flat.append(v.view(-1))
    if not flat:
        return torch.tensor([], device=device)
    return torch.cat(flat, dim=0)


def _vector_to_state_dict(vec: torch.Tensor, template_state_dict: dict, device: torch.device) -> dict:
    """
    Map a 1D tensor back to a state_dict, following template_state_dict order.

    Only trainable float parameters are overwritten; buffers (e.g., BN running stats) remain from template.
    """
    out = copy.deepcopy(template_state_dict)
    pointer = 0
    for k, param in out.items():
        if not _is_param_key(k, param):
            continue
        num = param.numel()
        slice_v = vec[pointer : pointer + num]
        out[k] = slice_v.view_as(param).to(device).type_as(param)
        pointer += num
    return out


def scope_defense(
    w_locals: List[dict],
    w_glob: dict,
    net_template: torch.nn.Module,
    args,
    client_lens: Optional[List[int]] = None,
) -> dict:
    """
    Scope defence (full algorithm adapted from Scope-main):
      - Flatten each client model and the global model.
      - Build per-dimension normalized signed differences (pre_metric) against the global model.
      - Compute cosine distances in this transformed space.
      - Run Dominant Gradient Clustering (DGC): from the most "central" client, iteratively follow nearest neighbours.
      - Aggregate only the clients on this chain, weighted by their data sizes.

    Inputs:
      w_locals:  list[state_dict] local client models for this round.
      w_glob:    global model state_dict before aggregation.
      net_template: unused, kept for API symmetry with other defences.
      args:      provides args.device if available.
      client_lens: optional list[int] of data sizes for weighting.

    Output:
      new global state_dict after Scope aggregation.
    """
    if len(w_locals) == 0:
        return w_glob

    device = getattr(args, "device", torch.device("cpu"))
    n_clients = len(w_locals)

    # flatten client models and global model following global parameter order
    vectorize_nets = []
    for sd in w_locals:
        vec = _flatten_state_dict(sd, w_glob, device=device)
        vectorize_nets.append(vec.detach().cpu().numpy().astype(np.float64))

    global_vec = _flatten_state_dict(w_glob, w_glob, device=device)
    global_vec_np = global_vec.detach().cpu().numpy().astype(np.float64)
    if global_vec_np.size == 0:
        # no trainable parameters; keep global weights
        return w_glob

    # -------- Dimension-wise normalized signed differences (pre_metric) --------
    pre_metric_dis = []
    eps = 1e-12
    abs_global = np.abs(global_vec_np)
    for g_i in vectorize_nets:
        diff = g_i - global_vec_np
        denom = np.abs(g_i) + abs_global
        # avoid division-by-zero on inactive dimensions while keeping behaviour close to original
        denom = np.where(denom == 0.0, eps, denom)
        pre_metric = np.power(np.abs(diff) / denom, 2.0) * np.sign(diff)
        pre_metric_dis.append(pre_metric)

    # -------- Cosine distance matrix in Scope space --------
    cos_dis = np.zeros((n_clients, n_clients), dtype=np.float64)
    sum_dis = np.zeros(n_clients, dtype=np.float64)
    for i in range(n_clients):
        g_i = pre_metric_dis[i]
        norm_i = np.linalg.norm(g_i)
        for j in range(i, n_clients):
            g_j = pre_metric_dis[j]
            norm_j = np.linalg.norm(g_j)
            if norm_i < eps or norm_j < eps:
                d = 100.0
            else:
                val = 1.0 - float(np.dot(g_i, g_j) / (norm_i * norm_j))
                if abs(val) < 1e-6:
                    val = 100.0
                d = val
            cos_dis[i, j] = d
            cos_dis[j, i] = d
            sum_dis[i] += d
            if j != i:
                sum_dis[j] += d

    # -------- Dominant Gradient Clustering (chain of nearest neighbours) --------
    choice = int(np.argmin(sum_dis))
    cluster = [choice]
    for _ in range(n_clients - 1):
        # find nearest neighbour of current choice in Scope space
        tmp = int(np.argmin(cos_dis[choice]))
        if tmp not in cluster:
            cluster.append(tmp)
            choice = tmp
        else:
            break

    if len(cluster) == 0:
        cluster = list(range(n_clients))

    # -------- Data-size-weighted aggregation over selected cluster --------
    if client_lens is not None and len(client_lens) == n_clients:
        num_dps = np.asarray(client_lens, dtype=np.float64)
    else:
        num_dps = np.ones(n_clients, dtype=np.float64)

    selected_num_dps = num_dps[cluster]
    total = float(selected_num_dps.sum())
    if total <= 0.0:
        weights = np.ones(len(cluster), dtype=np.float64) / float(len(cluster))
    else:
        weights = selected_num_dps / total

    stacked = np.stack(vectorize_nets, axis=0)
    aggregated_vec = np.average(stacked[cluster, :], weights=weights, axis=0).astype(np.float32)
    aggregated_tensor = torch.from_numpy(aggregated_vec).to(device)

    # Use the first client's full state_dict as template (including BN stats),
    # and overwrite only trainable parameters with the aggregated vector.
    template_state = w_locals[0]
    new_w_glob = _vector_to_state_dict(aggregated_tensor, template_state, device=device)
    return new_w_glob


import copy
import torch
import numpy as np


def _flatten_update(update_dict):
    """
    Flatten a (delta) state_dict into a 1D tensor.

    AlignIns 官方代码是将整个 state_dict 扁平化后打分，这里保持一致：
    - 包含 BatchNorm 的 running_mean / running_var。
    - 也包含 num_batches_tracked 计数。
    """
    vec = []
    for key, param in update_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)


def _vector_to_state_dict(vec, template_state_dict):
    """Map 1D tensor back to dict following template_state_dict order."""
    state_dict = copy.deepcopy(template_state_dict)
    pointer = 0
    for key, param in state_dict.items():
        num_param = param.numel()
        state_dict[key] = vec[pointer : pointer + num_param].view_as(param).type_as(param)
        pointer += num_param
    return state_dict


def alignins_defense(w_updates, w_glob, args, client_lens=None, client_ids=None, attack_flags=None):
    """
    AlignIns aggregation (Direction Alignment Inspection).

    Inputs:
        w_updates: list of local updates (state_dict delta) for this round.
        w_glob:    current global model state_dict.
        args:      should provide alignins_sparsity, lambda_s, lambda_c.

    Output:
        new global state_dict after AlignIns aggregation.
    """
    device = args.device if hasattr(args, "device") else "cpu"

    if len(w_updates) == 0:
        return w_glob

    # flatten updates and global model
    flat_updates = [_flatten_update(u).to(device) for u in w_updates]
    flat_global = _flatten_update(w_glob).to(device)
    inter_model_updates = torch.stack(flat_updates, dim=0)

    # majority sign over all updates
    major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
    sparsity = getattr(args, "alignins_sparsity", 0.25)
    k = max(1, int(len(flat_updates[0]) * sparsity))
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    mpsa_list = []
    tda_list = []
    for upd in inter_model_updates:
        _, idx_topk = torch.topk(torch.abs(upd), k)
        # Matching Principal Sign Alignment on top-k entries
        mpsa = (torch.sum(torch.sign(upd[idx_topk]) == major_sign[idx_topk]) / idx_topk.numel()).item()
        mpsa_list.append(mpsa)
        # Total Direction Alignment (cosine with global direction)
        tda_list.append(cos(upd, flat_global).item())

    # robust z-score (median / std) for MPSA & TDA
    mpsa_arr = np.array(mpsa_list)
    tda_arr = np.array(tda_list)
    eps = 1e-12

    mpsa_std = np.std(mpsa_arr) + eps
    mpsa_med = np.median(mpsa_arr)
    mz_mpsa = np.abs(mpsa_arr - mpsa_med) / mpsa_std

    tda_std = np.std(tda_arr) + eps
    tda_med = np.median(tda_arr)
    mz_tda = np.abs(tda_arr - tda_med) / tda_std

    lambda_s = getattr(args, "lambda_s", 1.0)
    lambda_c = getattr(args, "lambda_c", 1.0)

    benign_idx = list(
        set(np.argwhere(mz_mpsa < lambda_s).flatten().tolist())
        .intersection(set(np.argwhere(mz_tda < lambda_c).flatten().tolist()))
    )

    total_clients = len(w_updates)
    selected_clients = len(benign_idx)

    # expose selection info for outer-loop checks
    args.alignins_selected_indices = benign_idx
    if client_ids is not None and len(client_ids) == total_clients:
        args.alignins_selected_client_ids = [int(client_ids[i]) for i in benign_idx]
    else:
        args.alignins_selected_client_ids = None
    if attack_flags is not None and len(attack_flags) == total_clients:
        args.alignins_selected_attack_flags = [bool(attack_flags[i]) for i in benign_idx]
    else:
        args.alignins_selected_attack_flags = None

    # basic diagnostics for potential false negatives / all filtered
    print(
        f"[AlignIns] selected {selected_clients}/{total_clients} updates after MPSA/TDA filtering "
        f"(lambda_s={lambda_s}, lambda_c={lambda_c}, sparsity={sparsity})"
    )
    if getattr(args, "debug_alignins", 0) == 1 and client_ids is not None and len(client_ids) == total_clients:
        selected_client_ids = [int(client_ids[i]) for i in benign_idx]
        print(f"[AlignIns][debug] round participants(client_ids): {list(map(int, client_ids))}")
        print(f"[AlignIns][debug] selected indices: {benign_idx}")
        print(f"[AlignIns][debug] selected client_ids: {selected_client_ids}")
        if attack_flags is not None and len(attack_flags) == total_clients:
            attack_selected = sum(bool(attack_flags[i]) for i in benign_idx)
            attack_total = sum(bool(x) for x in attack_flags)
            print(f"[AlignIns][debug] selected attack-generated updates: {attack_selected}/{attack_total}")

    if selected_clients == 0:
        print("[AlignIns][warn] all updates were filtered; fallback to keeping previous global model.")
        return w_glob

    benign_updates = torch.stack([inter_model_updates[i] for i in benign_idx], dim=0)

    # Post-filtering model clipping (follow official code spirit):
    # - compute clipping threshold from selected (benign_idx)
    # - clip ALL updates with this threshold
    selected_norms = torch.norm(benign_updates, dim=1, keepdim=True)
    norm_clip = selected_norms.median(dim=0)[0].item()
    all_updates = inter_model_updates
    all_norms = torch.norm(all_updates, dim=1, keepdim=True)
    all_norms_clipped = torch.clamp(all_norms, min=0.0, max=norm_clip)
    clipped_updates = all_updates / (all_norms + eps) * all_norms_clipped

    # Aggregate only selected indices (weighted if client_lens provided)
    selected_updates = torch.stack([clipped_updates[i] for i in benign_idx], dim=0)
    if client_lens is not None and len(client_lens) == len(w_updates):
        w = torch.tensor([client_lens[i] for i in benign_idx], dtype=selected_updates.dtype, device=device)
        w = w / (w.sum() + eps)
        aggregated_update = torch.sum(selected_updates * w.view(-1, 1), dim=0)
    else:
        aggregated_update = selected_updates.mean(dim=0)

    # apply update to global weights
    new_flat = flat_global + aggregated_update
    new_w_glob = _vector_to_state_dict(new_flat, w_glob)
    return new_w_glob

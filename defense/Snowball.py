import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_, kaiming_normal_
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# 仅跳过计数器，保留 BN 统计量用于检测
_SKIP_SUFFIX = {"num_batches_tracked"}


def _is_param_key(key: str) -> bool:
    suffix = key.split(".")[-1]
    return suffix not in _SKIP_SUFFIX


def _base_aggregate(model_updates, weight_aggregation):
    """Weighted average of update dicts; weights are list of floats/ints."""
    update_avg = copy.deepcopy(model_updates[0])
    w = np.array(weight_aggregation, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)
    for key in update_avg.keys():
        update_avg[key] = update_avg[key] * w[0]
        for i in range(1, len(model_updates)):
            update_avg[key] += model_updates[i][key] * w[i]
    return update_avg


def _cluster(init_ids, data):
    # 与官方实现一致：以给定 init_ids 为初始中心，簇数不超过样本数
    n_samples = len(data)
    if n_samples == 0:
        return np.zeros(0, dtype=int)
    n_clusters = max(1, min(len(init_ids), n_samples))
    init_pts = [data[int(i)] for i in init_ids[:n_clusters]]
    clusterer = KMeans(n_clusters=n_clusters, init=init_pts, n_init=1)
    return clusterer.fit_predict(data)


def _flatten_model(model_update, layer_patterns, ignore_substr=None):
    keys = []
    for k in model_update.keys():
        if ignore_substr is not None and ignore_substr in k:
            continue
        for p in layer_patterns:
            if p in k:
                keys.append(k)
                break
    if len(keys) == 0:
        # fallback: flatten everything
        keys = list(model_update.keys())
    return torch.cat([model_update[k].flatten() for k in keys])


def _init_weights(model, init_type):
    if init_type not in ["none", "xavier", "kaiming"]:
        raise ValueError('init must be one of "none", "xavier" or "kaiming"')

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "xavier":
                xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                kaiming_normal_(m.weight.data, nonlinearity="relu")

    if init_type != "none":
        model.apply(init_func)


def _build_dif_set(data):
    dif_set = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                dif_set.append(data[i] - data[j])
    return dif_set


def _obtain_dif(base, target):
    dif_set = []
    for item in base:
        if torch.sum(item - target) != 0.0:
            dif_set.append(item - target)
            dif_set.append(target - item)
    return dif_set


kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = torch.nn.MSELoss(reduction="sum")


class MyDST(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e1(x_in.view(-1, self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus(self.fc_logvar(x))
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = torch.sigmoid(self.fc_d3(z))
        return x_out.view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn_like(sd))
        return e.mul(sd).add_(mean)

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

    def recon_prob(self, x_in, L=10):
        with torch.no_grad():
            x_in = torch.unsqueeze(x_in, dim=0)
            x_in = torch.sigmoid(x_in)
            mean, log_var = self.encoder(x_in)
            samples_z = [self.sample_normal(mean, log_var) for _ in range(L)]
            reconstruction_prob = 0.0
            for z in samples_z:
                x_logit = self.decoder(z)
                reconstruction_prob += recon_loss(x_logit, x_in).item()
            return reconstruction_prob / L


def _train_vae(vae, data, num_epoch, device, latent, hidden):
    data = torch.stack(data, dim=0)
    data = torch.sigmoid(data)
    if vae is None:
        vae = VAE(input_dim=len(data[0]), latent_dim=latent, hidden_dim=hidden).to(device)
        _init_weights(vae, "kaiming")
    vae = vae.to(device)
    vae.train()
    train_loader = DataLoader(MyDST(data), batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters())
    for _ in range(num_epoch):
        for x in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            recon_x, mu, logvar = vae(x)
            recon = recon_loss(recon_x, x)
            kl = torch.mean(kl_loss(mu, logvar))
            loss = recon + kl
            loss.backward()
            optimizer.step()
    return vae.cpu()


def _default_layer_patterns(args):
    # 针对本仓库的命名，覆盖更多 ResNet/VGG 中间层，避免只看首末层导致信号缺失
    m = getattr(args, "model", "")
    if m in {"cnn", "rlr_mnist"}:
        return ["conv1", "fc2"]
    if m in {"resnet", "resnet20"}:
        return ["conv1", "bn1", "layer1", "layer2", "layer3", "linear"]
    if m in {"VGG", "VGG11"}:
        return ["features.", "classifier."]
    return ["conv1", "fc"]


def _warmup_rounds(args):
    # 适配官方 Snowball-main 的数据集预热轮数
    if getattr(args, "snowball_warmup", -1) >= 0:
        return int(args.snowball_warmup)
    ds = getattr(args, "dataset", "")
    if ds in {"mnist"}:
        return 25
    if ds in {"fashion_mnist", "fashionmnist", "fmnist"}:
        return 30
    if ds in {"cifar", "cifar10", "cifar-10"}:
        return 100
    if ds in {"femnist"}:
        return 80
    if ds in {"sent140", "reddit"}:
        return 30
    return 30


def snowball_defense(w_updates, w_glob, args, client_lens=None, cur_round=0):
    """
    Snowball defense (AAAI 2024): bidirectional elections from individual perspective.

    Inputs:
      - w_updates: list[state_dict] of client deltas (local - global)
      - w_glob: global state_dict (for applying the aggregated update)
      - args: uses snowball_ct, snowball_vt, snowball_v_step, snowball_vae_* and model/dataset/gpu/device
      - client_lens: optional weights for aggregation
      - cur_round: current round index (0-based ok; compared against warmup threshold)

    Output:
      - new global state_dict
    """
    if len(w_updates) == 0:
        return w_glob

    # ---- hyperparameters ----
    ct = getattr(args, "snowball_ct", 10)
    vt = getattr(args, "snowball_vt", 0.5)
    v_step = getattr(args, "snowball_v_step", 0.05)
    vae_hidden = getattr(args, "snowball_vae_hidden", 256)
    vae_latent = getattr(args, "snowball_vae_latent", 64)
    vae_initial = getattr(args, "snowball_vae_initial", 270)
    vae_tuning = getattr(args, "snowball_vae_tuning", 30)

    layer_patterns = _default_layer_patterns(args)
    if getattr(args, "snowball_layers", ""):
        layer_patterns = [x.strip() for x in str(args.snowball_layers).split(",") if x.strip()]

    device = getattr(args, "device", torch.device("cpu"))

    # ---- filter to parameter-like keys only (avoid BN buffers / int counters) ----
    model_updates = []
    for upd in w_updates:
        d = {}
        for k, v in upd.items():
            if not _is_param_key(k):
                continue
            if not torch.is_floating_point(v):
                continue
            d[k] = v.detach().cpu()
        model_updates.append(d)

    if len(model_updates) == 0 or len(model_updates[0]) == 0:
        return w_glob

    weight_aggregation = (
        [int(x) for x in client_lens]
        if client_lens is not None and len(client_lens) == len(model_updates)
        else [1.0 for _ in range(len(model_updates))]
    )
    idx_list = list(range(len(model_updates)))

    # ---- Bottom-Up Election ----
    kernels = []
    for key in model_updates[0].keys():
        kernels.append([model_updates[i][key] for i in range(len(model_updates))])

    cnt = [0.0 for _ in range(len(model_updates))]
    for idx_layer, layer_name in enumerate(model_updates[0].keys()):
        if not any(p in layer_name for p in layer_patterns):
            continue

        benign_list_cur_layer = []
        score_list_cur_layer = []
        updates_kernel = [item.flatten().numpy() for item in kernels[idx_layer]]
        for idx_client in range(len(updates_kernel)):
            ddif = [updates_kernel[idx_client] - updates_kernel[i] for i in range(len(updates_kernel))]
            # if all identical, skip to avoid zero-var clustering issues
            if len(ddif) < 2 or np.allclose(ddif, 0):
                continue
            norms = np.linalg.norm(ddif, axis=1)
            norm_rank = np.argsort(norms)
            # 官方直接使用 ct（即 1+ct 个中心），小样本下可能失效
            suspicious_idx = norm_rank[-int(ct):] if int(ct) > 0 else []
            centroid_ids = [idx_client] + list(suspicious_idx)
            # dedupe to avoid repeated centroids
            centroid_ids = list(dict.fromkeys([int(x) for x in centroid_ids]))
            cluster_result = _cluster(centroid_ids, ddif)
            # CH score only valid when 2 <= labels <= n_samples-1
            n_labels = len(np.unique(cluster_result))
            if n_labels < 2 or n_labels >= len(ddif):
                score_ = 0.0
            else:
                score_ = calinski_harabasz_score(ddif, cluster_result)
            benign_ids = np.argwhere(cluster_result == cluster_result[idx_client]).flatten()

            benign_list_cur_layer.append(benign_ids)
            score_list_cur_layer.append(score_)

        score_list_cur_layer = np.array(score_list_cur_layer)
        effective_ids = np.argwhere(score_list_cur_layer > 0).flatten()
        if len(effective_ids) < int(len(score_list_cur_layer) * 0.1):
            effective_ids = np.argsort(-score_list_cur_layer)[: max(1, int(len(score_list_cur_layer) * 0.1))]

        if len(score_list_cur_layer) == 0:
            continue
        denom = (np.max(score_list_cur_layer) - np.min(score_list_cur_layer)) + 1e-12
        score_list_cur_layer = (score_list_cur_layer - np.min(score_list_cur_layer)) / denom
        for idx_client in effective_ids:
            for idx_b in benign_list_cur_layer[idx_client]:
                cnt[int(idx_b)] += float(score_list_cur_layer[idx_client])

    if len(cnt) == 0 or np.allclose(cnt, 0):
        # 避免因客户端顺序偏置（恶意客户端在前）导致默认选择攻击者，使用随机选取
        sel_size = max(1, int(len(model_updates) * 0.1))
        selected_ids = np.random.choice(len(model_updates), sel_size, replace=False).tolist()
    else:
        cnt_rank = np.argsort(-np.array(cnt))
        selected_ids = cnt_rank[: math.ceil(len(cnt_rank) * 0.1)].tolist()

    # ---- Warmup or insufficient seed: only bottom-up selection ----
    if int(cur_round) < _warmup_rounds(args) or len(selected_ids) < 2:
        update_avg = _base_aggregate([model_updates[i] for i in selected_ids], [weight_aggregation[i] for i in selected_ids])
    else:
        # ---- Top-Down Election (VAE) ----
        flatten_update_list = [_flatten_model(u, layer_patterns=layer_patterns) for u in model_updates]
        seed_data = _build_dif_set([flatten_update_list[i] for i in selected_ids])
        if len(seed_data) == 0:
            update_avg = _base_aggregate([model_updates[i] for i in selected_ids], [weight_aggregation[i] for i in selected_ids])
        else:
            vae = _train_vae(
                None,
                seed_data,
                vae_initial,
                device=device,
                latent=vae_latent,
                hidden=vae_hidden,
            )
            while len(selected_ids) < int(len(idx_list) * vt):
                seed_data = _build_dif_set([flatten_update_list[i] for i in selected_ids])
                if len(seed_data) == 0:
                    break
                vae = _train_vae(
                    vae,
                    seed_data,
                    vae_tuning,
                    device=device,
                    latent=vae_latent,
                    hidden=vae_hidden,
                )
                vae.eval()
                with torch.no_grad():
                    rest_ids = [i for i in range(len(flatten_update_list)) if i not in selected_ids]
                    loss_ = []
                    for idx in rest_ids:
                        m_loss = 0.0
                        loss_cnt = 0
                        dif_list = _obtain_dif([flatten_update_list[i] for i in selected_ids], flatten_update_list[idx])
                        if len(dif_list) == 0:
                            loss_.append(float("inf"))
                            continue
                        for dif in dif_list:
                            m_loss += float(vae.recon_prob(dif))
                            loss_cnt += 1
                        m_loss /= max(1, loss_cnt)
                        loss_.append(m_loss)
                rank_ = np.argsort(loss_)
                step = min(math.ceil(len(idx_list) * v_step), int(len(idx_list) * vt) - len(selected_ids))
                if step <= 0 or len(rest_ids) == 0:
                    break
                selected_ids.extend(np.array(rest_ids)[rank_[:step]].tolist())

            update_avg = _base_aggregate([model_updates[i] for i in selected_ids], [weight_aggregation[i] for i in selected_ids])

    if getattr(args, "debug", 0):
        print(
            f"[Snowball] selected {len(selected_ids)}/{len(model_updates)} clients for aggregation "
            f"(warmup<{_warmup_rounds(args)})"
        )

    # ---- Apply update_avg to global weights (only keys we aggregated) ----
    new_w_glob = copy.deepcopy(w_glob)
    for k, delta in update_avg.items():
        if k not in new_w_glob:
            continue
        new_w_glob[k] = new_w_glob[k] + delta.to(new_w_glob[k].device).type_as(new_w_glob[k])
    return new_w_glob

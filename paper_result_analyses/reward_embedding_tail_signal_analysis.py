"""Optional large-sample reward-embedding tail analysis.

This script extends the main reward-embedding/UMAP analysis to larger shared
seed banks and explicitly stratifies by reward tails. It is exploratory and is
not the primary paper-backed UMAP entrypoint; that role belongs to
`reward_embedding_tsne_analysis.py`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    reported_run_dir,
    reward_embedding_analysis_dir,
)

for _path in (REPO_ROOT, EG3D_ROOT, RLHF_CORE_ROOT.parent):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

from core_modules.data.create_train_data import generation_utils as gen_utils
from core_modules.utils import finetuning_utils
from core_modules.utils import reward_loading


@dataclass
class TailSignalConfig:
    reward_model_id: str = "7wnzkgie"
    tuned_pkl: str = str(reported_run_dir() / "network-snapshot-002068_LAST.pkl")
    results_dir: str = str(
        reward_embedding_analysis_dir() / "tail_signal_large_shared_seeds"
    )
    shape_res: int = 256
    truncation_cutoff: int = 14
    truncations: Tuple[float, float] = (0.7, 1.0)
    shared_seed_start: int = 700000
    n_samples: int = 2000
    tail_k: int = 100
    umap_n_neighbors: int = 25
    umap_min_dist: float = 0.1
    umap_random_state: int = 42
    pca_components: int = 50
    linear_probe_pca_components: int = 32
    knn_purity_k: int = 10
    max_batch: int = 1_000_000
    device: str = "cuda"
    noise_mode: str = "const"


def _to_serializable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    return x


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def _seed_to_z(seed: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    z_np = np.random.RandomState(int(seed)).randn(512).astype(np.float32)
    z = torch.from_numpy(z_np).unsqueeze(0).to(device)
    return z, z_np


def _z_pca1(z_codes: np.ndarray) -> np.ndarray:
    if len(z_codes) < 2:
        return np.zeros(len(z_codes), dtype=np.float32)
    reducer = PCA(n_components=1, random_state=0)
    return reducer.fit_transform(z_codes).reshape(-1).astype(np.float32)


def _load_reward_assets(cfg: TailSignalConfig, device: torch.device):
    model = reward_loading.load_rwd_model_from_cfg(cfg.reward_model_id)
    run_config_path = (
        RLHF_CORE_ROOT / "RWD_MODELS_FOR_TUNING" / cfg.reward_model_id / "run_config.yaml"
    )
    run_config = OmegaConf.load(run_config_path)
    sigma_aug = hydra.utils.instantiate(run_config.data.augmentations.sigma_norm)
    sigma_aug.eval()
    return model.to(device).eval(), sigma_aug


def _load_generator_assets(network_pkl: str, truncation_psi: float, shape_res: int):
    da = gen_utils.load_generator(
        model_path=Path(network_pkl),
        truncation_psi=truncation_psi,
        truncation_cutoff=14,
        shape_res=shape_res,
    )
    cond_path = gen_utils.STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt"
    cond = torch.load(cond_path, map_location=da.device)
    pads_path = gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml"
    pads = OmegaConf.load(pads_path)
    return da, cond, pads


def _prepare_sigma_sampling(
    G,
    pads_vals,
    shape_res: int,
) -> Tuple[finetuning_utils.MeshUtilsDataClass, torch.Tensor, Tuple[int, ...]]:
    mudc = finetuning_utils.MeshUtilsDataClass()
    samples, shape, _ = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads_vals,
        G=G,
        shape_res=shape_res,
    )
    return mudc, samples, shape


def _sample_sigma_volume(
    mudc: finetuning_utils.MeshUtilsDataClass,
    G,
    conditioning_params: torch.Tensor,
    samples: torch.Tensor,
    shape: Tuple[int, ...],
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    noise_mode: str,
    max_batch: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    z, z_np = _seed_to_z(seed, device=device)
    with torch.no_grad():
        sigmas = mudc.mesh_subset_of_points_from_samples_from_z_with_grad(
            G=G,
            z=z,
            conditioning_params=conditioning_params,
            samples=samples,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            update_emas=False,
            max_batch=max_batch,
        )
    volume = sigmas.squeeze(0).squeeze(-1).reshape(shape[1:4]).detach().cpu().float()
    return volume, z_np


def _reward_forward(
    reward_model,
    sigma_aug,
    volume_xyz: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    aug_volume = sigma_aug(volume_xyz.clone())
    model_input = aug_volume.permute(2, 1, 0).contiguous().unsqueeze(0).to(device)
    with torch.no_grad():
        emb8192 = reward_model.Conv3DModule.forward_to_global_vec(
            model_input, return_global_only=True
        )
        emb512 = reward_model.MLP(emb8192)
        reward = reward_model.forward_to_scalar_reward_from_single_global(emb512)
    return (
        emb8192.squeeze(0).detach().cpu().numpy().astype(np.float32),
        emb512.squeeze(0).detach().cpu().numpy().astype(np.float32),
        float(reward.squeeze().detach().cpu().item()),
    )


def _compute_umap(
    features: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    pca_components: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_samples, n_features = features.shape
    if n_samples < 3:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        return coords, {"used_pca_components": 0, "effective_n_neighbors": 0, "min_dist": min_dist}

    n_pca = min(pca_components, n_samples - 1, n_features)
    if n_pca >= 2 and n_features > n_pca:
        reducer = PCA(n_components=n_pca, random_state=random_state)
        reducer_input = reducer.fit_transform(features)
        pca_var = float(np.sum(reducer.explained_variance_ratio_))
    else:
        reducer_input = features
        pca_var = 1.0
        n_pca = features.shape[1]

    effective_n_neighbors = int(min(max(2, n_neighbors), max(2, n_samples - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init="spectral",
    )
    coords = reducer.fit_transform(reducer_input).astype(np.float32)
    meta = {
        "used_pca_components": int(n_pca),
        "pca_explained_variance_sum": pca_var,
        "effective_n_neighbors": effective_n_neighbors,
        "min_dist": min_dist,
    }
    return coords, meta


def _plot_scatter_matplotlib(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(9, 7))
    series = df[color_col]
    if pd.api.types.is_numeric_dtype(series):
        sc = plt.scatter(df[x], df[y], c=series, cmap="viridis", s=38, alpha=0.9)
        plt.colorbar(sc, label=color_col)
    else:
        cats = pd.Categorical(series)
        cmap = plt.get_cmap("tab10")
        for idx, cat in enumerate(cats.categories):
            mask = series == cat
            plt.scatter(df.loc[mask, x], df.loc[mask, y], s=38, alpha=0.9, label=str(cat), color=cmap(idx % 10))
        plt.legend(frameon=False, fontsize=9)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_with_plotly(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str,
    out_path: Path,
    title: str,
    hover_cols: Sequence[str],
) -> None:
    try:
        import plotly.express as px
    except Exception:
        return
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color_col,
        hover_data=list(hover_cols),
        title=title,
        opacity=0.85,
    )
    fig.update_traces(marker={"size": 8})
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _safe_silhouette(features: np.ndarray, labels: Sequence[Any]) -> float | None:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(features) <= len(np.unique(labels)):
        return None
    return float(silhouette_score(features, labels))


def _knn_purity(features: np.ndarray, labels: Sequence[str], k: int) -> float | None:
    labels = np.asarray(labels)
    n = len(labels)
    if n <= 2:
        return None
    k = int(min(max(1, k), n - 1))
    dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    same = labels[nn_idx] == labels[:, None]
    return float(same.mean())


def _linear_probe_auc(features: np.ndarray, labels: Sequence[str], pca_components: int) -> float | None:
    y = (np.asarray(labels) == "high").astype(np.int64)
    if len(np.unique(y)) < 2:
        return None
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return None
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=min(pca_components, features.shape[0] - 1, features.shape[1]), random_state=0)),
            ("clf", LogisticRegression(max_iter=5000, solver="liblinear")),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    probs = cross_val_predict(pipeline, features, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, probs))


def _reward_pc1_corr(features: np.ndarray, rewards: Sequence[float]) -> Dict[str, float]:
    if len(features) < 2:
        return {"pearson": 0.0, "spearman": 0.0}
    pc1 = PCA(n_components=1, random_state=0).fit_transform(features).reshape(-1)
    reward_series = pd.Series(np.asarray(rewards, dtype=np.float32))
    pc1_series = pd.Series(pc1.astype(np.float32))
    return {
        "pearson": float(reward_series.corr(pc1_series, method="pearson")),
        "spearman": float(reward_series.corr(pc1_series, method="spearman")),
    }


def _sample_regime(
    cfg: TailSignalConfig,
    reward_model,
    sigma_aug,
    device: torch.device,
    truncation_psi: float,
    out_dir: Path,
) -> Dict[str, Any]:
    seeds = np.arange(cfg.shared_seed_start, cfg.shared_seed_start + cfg.n_samples, dtype=np.int64)
    seed_table = pd.DataFrame(
        {
            "seed": seeds,
            "truncation_psi": float(truncation_psi),
            "regime_name": f"trunc_{truncation_psi:.2f}",
        }
    )

    da_tuned, cond, pads = _load_generator_assets(cfg.tuned_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc, samples, shape = _prepare_sigma_sampling(da_tuned.G, pads_vals=pads, shape_res=cfg.shape_res)

    records: List[Dict[str, Any]] = []
    emb8192_list: List[np.ndarray] = []
    emb512_list: List[np.ndarray] = []
    z_codes: List[np.ndarray] = []

    iterator = tqdm(seed_table.itertuples(index=False), total=len(seed_table), desc=f"psi={truncation_psi:.2f}")
    for row in iterator:
        volume_xyz, z_np = _sample_sigma_volume(
            mudc=mudc,
            G=da_tuned.G,
            conditioning_params=cond,
            samples=samples,
            shape=shape,
            seed=int(row.seed),
            truncation_psi=float(row.truncation_psi),
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        emb8192, emb512, reward = _reward_forward(
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            volume_xyz=volume_xyz,
            device=device,
        )
        emb8192_list.append(emb8192)
        emb512_list.append(emb512)
        z_codes.append(z_np)
        records.append(
            {
                "seed": int(row.seed),
                "truncation_psi": float(row.truncation_psi),
                "regime_name": str(row.regime_name),
                "reward_score": reward,
                "latent_l2_norm": float(np.linalg.norm(z_np)),
            }
        )

    full_df = pd.DataFrame.from_records(records)
    z_arr = np.stack(z_codes, axis=0).astype(np.float32)
    emb8192 = np.stack(emb8192_list, axis=0).astype(np.float32)
    emb512 = np.stack(emb512_list, axis=0).astype(np.float32)
    full_df["z_pca1"] = _z_pca1(z_arr)

    full_df.to_csv(out_dir / "full_samples.csv", index=False)
    _save_npz(out_dir / "full_embeddings.npz", emb8192=emb8192, emb512=emb512, z_codes=z_arr)

    ranked = full_df.sort_values("reward_score").reset_index(drop=True)
    tail_k = min(cfg.tail_k, len(ranked) // 2)
    low_idx = ranked.index[:tail_k].to_numpy()
    high_idx = ranked.index[-tail_k:].to_numpy()
    tail_idx = np.concatenate([low_idx, high_idx], axis=0)
    gather_idx = ranked.index.to_numpy()[tail_idx]

    tail_df = ranked.loc[tail_idx].copy().reset_index(drop=True)
    tail_df["tail_label"] = ["low"] * len(low_idx) + ["high"] * len(high_idx)
    tail8192 = emb8192[gather_idx]
    tail512 = emb512[gather_idx]
    tailz = z_arr[gather_idx]

    umap8192, meta8192 = _compute_umap(
        tail8192,
        cfg.umap_n_neighbors,
        cfg.umap_min_dist,
        cfg.umap_random_state,
        cfg.pca_components,
    )
    umap512, meta512 = _compute_umap(
        tail512,
        cfg.umap_n_neighbors,
        cfg.umap_min_dist,
        cfg.umap_random_state,
        cfg.pca_components,
    )
    tail_df["umap8192_x"] = umap8192[:, 0]
    tail_df["umap8192_y"] = umap8192[:, 1]
    tail_df["umap512_x"] = umap512[:, 0]
    tail_df["umap512_y"] = umap512[:, 1]

    tail_df.to_csv(out_dir / "tail_samples.csv", index=False)
    _save_npz(out_dir / "tail_embeddings.npz", emb8192=tail8192, emb512=tail512, z_codes=tailz)

    for color_col in ("tail_label", "reward_score", "z_pca1"):
        _plot_scatter_matplotlib(
            tail_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"umap8192_color_{color_col}.png",
            title=f"psi={truncation_psi:.2f} 8192 UMAP colored by {color_col}",
        )
        _plot_scatter_matplotlib(
            tail_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"umap512_color_{color_col}.png",
            title=f"psi={truncation_psi:.2f} 512 UMAP colored by {color_col}",
        )
        hover = ["seed", "reward_score", "tail_label", "z_pca1"]
        _plot_with_plotly(
            tail_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"umap8192_color_{color_col}.html",
            title=f"psi={truncation_psi:.2f} 8192 UMAP colored by {color_col}",
            hover_cols=hover,
        )
        _plot_with_plotly(
            tail_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"umap512_color_{color_col}.html",
            title=f"psi={truncation_psi:.2f} 512 UMAP colored by {color_col}",
            hover_cols=hover,
        )

    low_rewards = tail_df.loc[tail_df["tail_label"] == "low", "reward_score"]
    high_rewards = tail_df.loc[tail_df["tail_label"] == "high", "reward_score"]
    labels = tail_df["tail_label"].to_numpy()
    summary = {
        "n_samples": int(len(full_df)),
        "tail_k": int(tail_k),
        "shape": tuple(int(v) for v in shape[1:4]),
        "reward_gap_mean": float(high_rewards.mean() - low_rewards.mean()),
        "reward_low_mean": float(low_rewards.mean()),
        "reward_high_mean": float(high_rewards.mean()),
        "reward_full_mean": float(full_df["reward_score"].mean()),
        "reward_full_std": float(full_df["reward_score"].std()),
        "silhouette_tail_8192": _safe_silhouette(tail8192, labels),
        "silhouette_tail_512": _safe_silhouette(tail512, labels),
        "knn_purity_tail_8192": _knn_purity(tail8192, labels, cfg.knn_purity_k),
        "knn_purity_tail_512": _knn_purity(tail512, labels, cfg.knn_purity_k),
        "linear_probe_auc_tail_8192": _linear_probe_auc(tail8192, labels, cfg.linear_probe_pca_components),
        "linear_probe_auc_tail_512": _linear_probe_auc(tail512, labels, cfg.linear_probe_pca_components),
        "reward_pc1_corr_8192": _reward_pc1_corr(emb8192, full_df["reward_score"].to_numpy()),
        "reward_pc1_corr_512": _reward_pc1_corr(emb512, full_df["reward_score"].to_numpy()),
        "umap8192": meta8192,
        "umap512": meta512,
        "top_tail_seeds": tail_df.loc[tail_df["tail_label"] == "high", "seed"].tolist(),
        "bottom_tail_seeds": tail_df.loc[tail_df["tail_label"] == "low", "seed"].tolist(),
    }
    _write_json(out_dir / "summary.json", summary)
    return {
        "full_df": full_df,
        "tail_df": tail_df,
        "emb8192": emb8192,
        "emb512": emb512,
        "summary": summary,
    }


def main() -> None:
    cfg = TailSignalConfig()
    device = torch.device(cfg.device)
    out_dir = _ensure_dir(Path(cfg.results_dir))

    reward_model, sigma_aug = _load_reward_assets(cfg, device)

    regime_results: Dict[str, Dict[str, Any]] = {}
    for truncation_psi in cfg.truncations:
        regime_dir = _ensure_dir(out_dir / f"psi_{truncation_psi:.2f}".replace(".", "p"))
        regime_results[f"psi_{truncation_psi:.2f}"] = _sample_regime(
            cfg=cfg,
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            device=device,
            truncation_psi=truncation_psi,
            out_dir=regime_dir,
        )

    comparison = {}
    if len(cfg.truncations) == 2:
        k0 = f"psi_{cfg.truncations[0]:.2f}"
        k1 = f"psi_{cfg.truncations[1]:.2f}"
        s0 = regime_results[k0]["summary"]
        s1 = regime_results[k1]["summary"]
        comparison = {
            "shared_seed_start": int(cfg.shared_seed_start),
            "shared_n_samples": int(cfg.n_samples),
            "reward_gap_delta": float(s1["reward_gap_mean"] - s0["reward_gap_mean"]),
            "silhouette_8192_delta": float((s1["silhouette_tail_8192"] or 0.0) - (s0["silhouette_tail_8192"] or 0.0)),
            "silhouette_512_delta": float((s1["silhouette_tail_512"] or 0.0) - (s0["silhouette_tail_512"] or 0.0)),
            "knn_purity_8192_delta": float((s1["knn_purity_tail_8192"] or 0.0) - (s0["knn_purity_tail_8192"] or 0.0)),
            "knn_purity_512_delta": float((s1["knn_purity_tail_512"] or 0.0) - (s0["knn_purity_tail_512"] or 0.0)),
            "linear_probe_auc_8192_delta": float((s1["linear_probe_auc_tail_8192"] or 0.0) - (s0["linear_probe_auc_tail_8192"] or 0.0)),
            "linear_probe_auc_512_delta": float((s1["linear_probe_auc_tail_512"] or 0.0) - (s0["linear_probe_auc_tail_512"] or 0.0)),
        }

    summary = {
        "config": asdict(cfg),
        "regimes": {key: value["summary"] for key, value in regime_results.items()},
        "comparison": comparison,
    }
    _write_json(out_dir / "tail_signal_summary.json", summary)
    print(json.dumps({"results_dir": str(out_dir), "summary_path": str(out_dir / "tail_signal_summary.json")}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import gc
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm
import umap

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    DEFAULT_NOREWARD_RUN_NAME,
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    dataset_zip_path,
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

import hydra

from core_modules.data.create_train_data import generation_utils as gen_utils
from core_modules.utils import finetuning_utils
from core_modules.utils import reward_loading


@dataclass
class AnalysisConfig:
    reward_model_id: str = "7wnzkgie"
    orig_pkl: str = str(REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl")
    tuned_pkl: str = str(reported_run_dir() / "network-snapshot-002068_LAST.pkl")
    noreward_pkl: str = str(
        reported_run_dir(
            run_name=DEFAULT_NOREWARD_RUN_NAME,
            explicit_env="EG3D_RLHF_NOREWARD_RUN_DIR",
        )
        / "network-snapshot-002068_LAST.pkl"
    )
    results_dir: str = str(reward_embedding_analysis_dir())
    fid_results_dir: str = str(
        reported_run_dir() / "geometry_compare_results" / "fid_comparison"
    )
    dataset_path: str = str(dataset_zip_path())
    shape_res: int = 256
    truncation_cutoff: int = 14
    n_per_regime: int = 100
    tail_k: int = 20
    exp1_truncations: Tuple[float, float, float] = (0.25, 0.7, 1.0)
    seed_starts: Tuple[int, int, int] = (100000, 200000, 300000)
    exp2b_seed_start: int = 400000
    exp2b_n_samples: int = 200
    exp2b_truncation: float = 1.0
    identity_match_truncation: float = 0.7
    tuned_identity_bank_seed_start: int = 500000
    tuned_identity_bank_size: int = 5000
    orig_identity_query_seed_start: int = 600000
    orig_identity_query_size: int = 500
    identity_match_topk: int = 20
    identity_match_target_pairs: int = 200
    identity_match_min_cosine: float = 0.80
    identity_batch_size: int = 32
    identity_render_batch_size: int = 4
    rgb_metric_resolution: int = 128
    umap_n_neighbors: int = 25
    umap_min_dist: float = 0.1
    umap_random_state: int = 42
    pca_components: int = 50
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


def _make_seed_table(start_seed: int, n: int, truncation_psi: float, regime_name: str) -> pd.DataFrame:
    seeds = np.arange(start_seed, start_seed + n, dtype=np.int64)
    return pd.DataFrame(
        {
            "seed": seeds,
            "truncation_psi": float(truncation_psi),
            "regime_name": regime_name,
        }
    )


def _seed_to_z(seed: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    z_np = np.random.RandomState(int(seed)).randn(512).astype(np.float32)
    z = torch.from_numpy(z_np).unsqueeze(0).to(device)
    return z, z_np


def _load_reward_assets(cfg: AnalysisConfig, device: torch.device):
    model = reward_loading.load_rwd_model_from_cfg(cfg.reward_model_id)
    run_config_path = (
        RLHF_CORE_ROOT / "RWD_MODELS_FOR_TUNING" / cfg.reward_model_id / "run_config.yaml"
    )
    run_config = OmegaConf.load(run_config_path)
    sigma_aug = hydra.utils.instantiate(run_config.data.augmentations.sigma_norm)
    sigma_aug.eval()
    return model.to(device).eval(), run_config, sigma_aug


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


def _get_single_img_cam_bundle(da) -> Tuple[torch.Tensor, torch.Tensor]:
    camera_params, conditioning_params = gen_utils.get_triple_img_cams(da)
    return camera_params[1].unsqueeze(0), conditioning_params[1].unsqueeze(0)


def _expand_camera(camera_params: torch.Tensor, batch_size: int) -> torch.Tensor:
    if camera_params.shape[0] == batch_size:
        return camera_params
    if camera_params.shape[0] != 1:
        raise ValueError(f"camera batch mismatch: got {camera_params.shape[0]}, wanted 1 or {batch_size}")
    return camera_params.expand(batch_size, -1)


def _render_rgb_batch(
    da,
    z_batch: torch.Tensor,
    camera_params: torch.Tensor,
    conditioning_params: torch.Tensor,
    neural_rendering_resolution: int,
    truncation_psi: float,
    truncation_cutoff: int,
    noise_mode: str,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(z_batch), batch_size), desc=desc):
            z_mb = z_batch[start : start + batch_size]
            render_c_mb = _expand_camera(camera_params, len(z_mb))
            cond_c_mb = _expand_camera(conditioning_params, len(z_mb))
            ws = da.G.mapping(
                z_mb,
                cond_c_mb,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
            )
            rgb = da.G.synthesis(
                ws,
                render_c_mb,
                neural_rendering_resolution=neural_rendering_resolution,
                noise_mode=noise_mode,
            )["image"]
            if rgb.shape[-1] != neural_rendering_resolution:
                rgb = F.interpolate(
                    rgb,
                    size=(neural_rendering_resolution, neural_rendering_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            outs.append(rgb.detach().cpu())
    return torch.cat(outs, dim=0)


def _load_face_identity_model(device: torch.device):
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _compute_face_embeddings(
    model,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    embeds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(images), batch_size), desc=desc):
            mb = images[start : start + batch_size].to(device)
            if mb.shape[-1] != 160:
                mb = F.interpolate(mb, size=(160, 160), mode="bilinear", align_corners=False)
            emb = F.normalize(model(mb), dim=1)
            embeds.append(emb.detach().cpu())
    return torch.cat(embeds, dim=0)


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


def _z_pca1(z_codes: np.ndarray) -> np.ndarray:
    if len(z_codes) < 2:
        return np.zeros(len(z_codes), dtype=np.float32)
    reducer = PCA(n_components=1, random_state=0)
    return reducer.fit_transform(z_codes).reshape(-1).astype(np.float32)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(_to_serializable(payload), f, indent=2)


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
            msk = series == cat
            plt.scatter(df.loc[msk, x], df.loc[msk, y], s=38, alpha=0.9, label=str(cat), color=cmap(idx % 10))
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


def _pairwise_summary(df: pd.DataFrame, key: str, value: str) -> Dict[str, float]:
    pivot = df.pivot(index=key, columns="model_label", values=value)
    if not {"orig", "tuned"}.issubset(pivot.columns):
        return {}
    delta = (pivot["tuned"] - pivot["orig"]).dropna()
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "frac_positive": float((delta > 0).mean()),
        "frac_negative": float((delta < 0).mean()),
    }


def _safe_silhouette(features: np.ndarray, labels: Sequence[Any]) -> Optional[float]:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(features) <= len(np.unique(labels)):
        return None
    return float(silhouette_score(features, labels))


def _collect_features_for_table(
    table: pd.DataFrame,
    generator_label: str,
    G,
    conditioning_params: torch.Tensor,
    mudc: finetuning_utils.MeshUtilsDataClass,
    samples: torch.Tensor,
    shape: Tuple[int, ...],
    reward_model,
    sigma_aug,
    cfg: AnalysisConfig,
    device: torch.device,
    desc: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    records: List[Dict[str, Any]] = []
    emb8192_list: List[np.ndarray] = []
    emb512_list: List[np.ndarray] = []
    z_codes: List[np.ndarray] = []

    iterator = tqdm(table.itertuples(index=False), total=len(table), desc=desc)
    for row in iterator:
        volume_xyz, z_np = _sample_sigma_volume(
            mudc=mudc,
            G=G,
            conditioning_params=conditioning_params,
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
                "generator_label": generator_label,
                "reward_score": reward,
                "latent_l2_norm": float(np.linalg.norm(z_np)),
            }
        )

    df = pd.DataFrame.from_records(records)
    z_arr = np.stack(z_codes, axis=0).astype(np.float32)
    df["z_pca1"] = _z_pca1(z_arr)
    return (
        df,
        np.stack(emb8192_list, axis=0).astype(np.float32),
        np.stack(emb512_list, axis=0).astype(np.float32),
        z_arr,
    )


def _run_exp1(
    reward_model,
    sigma_aug,
    cfg: AnalysisConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    tables = []
    table_parts = [
        _make_seed_table(start_seed=s0, n=cfg.n_per_regime, truncation_psi=psi, regime_name=f"trunc_{psi:.2f}")
        for s0, psi in zip(cfg.seed_starts, cfg.exp1_truncations)
    ]
    seed_table = pd.concat(table_parts, ignore_index=True)

    da_tuned, cond, pads = _load_generator_assets(cfg.tuned_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc, samples, shape = _prepare_sigma_sampling(da_tuned.G, pads_vals=pads, shape_res=cfg.shape_res)

    df, emb8192, emb512, z_codes = _collect_features_for_table(
        table=seed_table,
        generator_label="tuned",
        G=da_tuned.G,
        conditioning_params=cond,
        mudc=mudc,
        samples=samples,
        shape=shape,
        reward_model=reward_model,
        sigma_aug=sigma_aug,
        cfg=cfg,
        device=device,
        desc="exp1/tuned_sigma_sampling",
    )
    tables.append(df)
    df = pd.concat(tables, ignore_index=True)

    umap8192, meta8192 = _compute_umap(emb8192, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    umap512, meta512 = _compute_umap(emb512, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    df["umap8192_x"] = umap8192[:, 0]
    df["umap8192_y"] = umap8192[:, 1]
    df["umap512_x"] = umap512[:, 0]
    df["umap512_y"] = umap512[:, 1]

    _save_npz(out_dir / "exp1_embeddings.npz", emb8192=emb8192, emb512=emb512, z_codes=z_codes)
    df.to_csv(out_dir / "exp1_samples.csv", index=False)

    for color_col in ("regime_name", "reward_score", "z_pca1", "latent_l2_norm"):
        _plot_scatter_matplotlib(
            df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"exp1_umap8192_color_{color_col}.png",
            title=f"Exp1 8192 UMAP colored by {color_col}",
        )
        _plot_scatter_matplotlib(
            df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"exp1_umap512_color_{color_col}.png",
            title=f"Exp1 512 UMAP colored by {color_col}",
        )
        hover = ["seed", "truncation_psi", "reward_score", "latent_l2_norm", "z_pca1"]
        _plot_with_plotly(
            df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"exp1_umap8192_color_{color_col}.html",
            title=f"Exp1 8192 UMAP colored by {color_col}",
            hover_cols=hover,
        )
        _plot_with_plotly(
            df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"exp1_umap512_color_{color_col}.html",
            title=f"Exp1 512 UMAP colored by {color_col}",
            hover_cols=hover,
        )

    regime_summary = (
        df.groupby("regime_name")["reward_score"]
        .agg(["mean", "std", "min", "median", "max"])
        .reset_index()
    )
    regime_summary.to_csv(out_dir / "exp1_regime_reward_summary.csv", index=False)

    summary = {
        "n_samples": int(len(df)),
        "shape": tuple(int(v) for v in shape[1:4]),
        "umap8192": meta8192,
        "umap512": meta512,
        "reward_by_regime": regime_summary.to_dict(orient="records"),
        "silhouette_regime_8192": _safe_silhouette(emb8192, df["regime_name"]),
        "silhouette_regime_512": _safe_silhouette(emb512, df["regime_name"]),
    }
    _write_json(out_dir / "exp1_summary.json", summary)
    return {
        "df": df,
        "emb8192": emb8192,
        "emb512": emb512,
        "z_codes": z_codes,
        "summary": summary,
    }


def _run_exp2(
    exp1: Dict[str, Any],
    cfg: AnalysisConfig,
    out_dir: Path,
) -> Dict[str, Any]:
    df = exp1["df"].copy()
    emb8192 = exp1["emb8192"]
    emb512 = exp1["emb512"]
    z_codes = exp1["z_codes"]

    trunc_mask = np.isclose(df["truncation_psi"].to_numpy(), 0.7)
    sub_df = df.loc[trunc_mask].copy().reset_index(drop=True)
    sub8192 = emb8192[trunc_mask]
    sub512 = emb512[trunc_mask]
    subz = z_codes[trunc_mask]

    ranked = sub_df.sort_values("reward_score").reset_index(drop=True)
    tail_k = min(cfg.tail_k, len(ranked) // 2)
    low_idx = ranked.index[:tail_k].to_numpy()
    high_idx = ranked.index[-tail_k:].to_numpy()
    tail_idx = np.concatenate([low_idx, high_idx], axis=0)

    tail_df = ranked.loc[tail_idx].copy().reset_index(drop=True)
    tail_df["tail_label"] = ["low"] * len(low_idx) + ["high"] * len(high_idx)
    tail8192 = sub8192[ranked.index.to_numpy()[tail_idx]]
    tail512 = sub512[ranked.index.to_numpy()[tail_idx]]
    tailz = subz[ranked.index.to_numpy()[tail_idx]]

    umap8192, meta8192 = _compute_umap(tail8192, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    umap512, meta512 = _compute_umap(tail512, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    tail_df["umap8192_x"] = umap8192[:, 0]
    tail_df["umap8192_y"] = umap8192[:, 1]
    tail_df["umap512_x"] = umap512[:, 0]
    tail_df["umap512_y"] = umap512[:, 1]

    tail_df.to_csv(out_dir / "exp2_tail_samples.csv", index=False)
    _save_npz(out_dir / "exp2_tail_embeddings.npz", emb8192=tail8192, emb512=tail512, z_codes=tailz)

    for color_col in ("tail_label", "reward_score", "z_pca1"):
        _plot_scatter_matplotlib(
            tail_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"exp2_umap8192_color_{color_col}.png",
            title=f"Exp2 8192 UMAP colored by {color_col}",
        )
        _plot_scatter_matplotlib(
            tail_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"exp2_umap512_color_{color_col}.png",
            title=f"Exp2 512 UMAP colored by {color_col}",
        )
        hover = ["seed", "reward_score", "tail_label", "z_pca1"]
        _plot_with_plotly(
            tail_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"exp2_umap8192_color_{color_col}.html",
            title=f"Exp2 8192 UMAP colored by {color_col}",
            hover_cols=hover,
        )
        _plot_with_plotly(
            tail_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"exp2_umap512_color_{color_col}.html",
            title=f"Exp2 512 UMAP colored by {color_col}",
            hover_cols=hover,
        )

    low_rewards = tail_df.loc[tail_df["tail_label"] == "low", "reward_score"]
    high_rewards = tail_df.loc[tail_df["tail_label"] == "high", "reward_score"]
    summary = {
        "tail_k": int(tail_k),
        "umap8192": meta8192,
        "umap512": meta512,
        "reward_gap_mean": float(high_rewards.mean() - low_rewards.mean()),
        "reward_low_mean": float(low_rewards.mean()),
        "reward_high_mean": float(high_rewards.mean()),
        "silhouette_tail_8192": _safe_silhouette(tail8192, tail_df["tail_label"]),
        "silhouette_tail_512": _safe_silhouette(tail512, tail_df["tail_label"]),
        "top_tail_seeds": tail_df.loc[tail_df["tail_label"] == "high", "seed"].tolist(),
        "bottom_tail_seeds": tail_df.loc[tail_df["tail_label"] == "low", "seed"].tolist(),
    }
    _write_json(out_dir / "exp2_summary.json", summary)
    return {
        "df": tail_df,
        "emb8192": tail8192,
        "emb512": tail512,
        "summary": summary,
    }


def _run_tail_analysis_from_table(
    df: pd.DataFrame,
    emb8192: np.ndarray,
    emb512: np.ndarray,
    z_codes: np.ndarray,
    tail_k: int,
    out_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    ranked = df.sort_values("reward_score").reset_index(drop=True)
    tail_k = min(tail_k, len(ranked) // 2)
    low_idx = ranked.index[:tail_k].to_numpy()
    high_idx = ranked.index[-tail_k:].to_numpy()
    tail_idx = np.concatenate([low_idx, high_idx], axis=0)

    tail_df = ranked.loc[tail_idx].copy().reset_index(drop=True)
    tail_df["tail_label"] = ["low"] * len(low_idx) + ["high"] * len(high_idx)
    gather_idx = ranked.index.to_numpy()[tail_idx]
    tail8192 = emb8192[gather_idx]
    tail512 = emb512[gather_idx]
    tailz = z_codes[gather_idx]

    umap8192, meta8192 = _compute_umap(tail8192, 25, 0.1, 42, min(50, tail8192.shape[1]))
    umap512, meta512 = _compute_umap(tail512, 25, 0.1, 42, min(50, tail512.shape[1]))
    tail_df["umap8192_x"] = umap8192[:, 0]
    tail_df["umap8192_y"] = umap8192[:, 1]
    tail_df["umap512_x"] = umap512[:, 0]
    tail_df["umap512_y"] = umap512[:, 1]

    tail_df.to_csv(out_dir / f"{prefix}_tail_samples.csv", index=False)
    _save_npz(out_dir / f"{prefix}_tail_embeddings.npz", emb8192=tail8192, emb512=tail512, z_codes=tailz)

    for color_col in ("tail_label", "reward_score", "z_pca1"):
        _plot_scatter_matplotlib(
            tail_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"{prefix}_umap8192_color_{color_col}.png",
            title=f"{prefix} 8192 UMAP colored by {color_col}",
        )
        _plot_scatter_matplotlib(
            tail_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"{prefix}_umap512_color_{color_col}.png",
            title=f"{prefix} 512 UMAP colored by {color_col}",
        )
        hover = ["seed", "reward_score", "tail_label", "z_pca1"]
        _plot_with_plotly(
            tail_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"{prefix}_umap8192_color_{color_col}.html",
            title=f"{prefix} 8192 UMAP colored by {color_col}",
            hover_cols=hover,
        )
        _plot_with_plotly(
            tail_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"{prefix}_umap512_color_{color_col}.html",
            title=f"{prefix} 512 UMAP colored by {color_col}",
            hover_cols=hover,
        )

    low_rewards = tail_df.loc[tail_df["tail_label"] == "low", "reward_score"]
    high_rewards = tail_df.loc[tail_df["tail_label"] == "high", "reward_score"]
    summary = {
        "tail_k": int(tail_k),
        "umap8192": meta8192,
        "umap512": meta512,
        "reward_gap_mean": float(high_rewards.mean() - low_rewards.mean()),
        "reward_low_mean": float(low_rewards.mean()),
        "reward_high_mean": float(high_rewards.mean()),
        "silhouette_tail_8192": _safe_silhouette(tail8192, tail_df["tail_label"]),
        "silhouette_tail_512": _safe_silhouette(tail512, tail_df["tail_label"]),
        "top_tail_seeds": tail_df.loc[tail_df["tail_label"] == "high", "seed"].tolist(),
        "bottom_tail_seeds": tail_df.loc[tail_df["tail_label"] == "low", "seed"].tolist(),
    }
    _write_json(out_dir / f"{prefix}_summary.json", summary)
    return {"df": tail_df, "emb8192": tail8192, "emb512": tail512, "summary": summary}


def _run_exp3(
    exp1: Dict[str, Any],
    reward_model,
    sigma_aug,
    cfg: AnalysisConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    tuned_df = exp1["df"].copy()
    tuned8192 = exp1["emb8192"]
    tuned512 = exp1["emb512"]
    tuned_z = exp1["z_codes"]

    trunc_mask = np.isclose(tuned_df["truncation_psi"].to_numpy(), 0.7)
    tuned_df = tuned_df.loc[trunc_mask].copy().reset_index(drop=True)
    tuned8192 = tuned8192[trunc_mask]
    tuned512 = tuned512[trunc_mask]
    tuned_z = tuned_z[trunc_mask]

    orig_seed_table = tuned_df[["seed", "truncation_psi", "regime_name"]].copy()
    da_orig, cond, pads = _load_generator_assets(cfg.orig_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc, samples, shape = _prepare_sigma_sampling(da_orig.G, pads_vals=pads, shape_res=cfg.shape_res)
    orig_df, orig8192, orig512, orig_z = _collect_features_for_table(
        table=orig_seed_table,
        generator_label="orig",
        G=da_orig.G,
        conditioning_params=cond,
        mudc=mudc,
        samples=samples,
        shape=shape,
        reward_model=reward_model,
        sigma_aug=sigma_aug,
        cfg=cfg,
        device=device,
        desc="exp3/orig_sigma_sampling",
    )

    tuned_df = tuned_df.copy()
    tuned_df["generator_label"] = "tuned"

    comb_df = pd.concat([orig_df, tuned_df], ignore_index=True)
    comb8192 = np.concatenate([orig8192, tuned8192], axis=0)
    comb512 = np.concatenate([orig512, tuned512], axis=0)
    comb_z = np.concatenate([orig_z, tuned_z], axis=0)
    comb_df["model_label"] = comb_df["generator_label"]

    umap8192, meta8192 = _compute_umap(comb8192, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    umap512, meta512 = _compute_umap(comb512, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    comb_df["umap8192_x"] = umap8192[:, 0]
    comb_df["umap8192_y"] = umap8192[:, 1]
    comb_df["umap512_x"] = umap512[:, 0]
    comb_df["umap512_y"] = umap512[:, 1]

    pair_reward = comb_df.pivot(index="seed", columns="model_label", values="reward_score")
    reward_delta_map = (pair_reward["tuned"] - pair_reward["orig"]).to_dict()
    comb_df["reward_delta_tuned_minus_orig"] = comb_df["seed"].map(reward_delta_map)

    pair_emb_l2 = {
        int(seed): float(np.linalg.norm(tuned512[i] - orig512[i]))
        for i, seed in enumerate(orig_df["seed"].tolist())
    }
    comb_df["embedding512_l2_delta"] = comb_df["seed"].map(pair_emb_l2)

    comb_df.to_csv(out_dir / "exp3_orig_vs_tuned_samples.csv", index=False)
    _save_npz(out_dir / "exp3_orig_vs_tuned_embeddings.npz", emb8192=comb8192, emb512=comb512, z_codes=comb_z)

    for color_col in ("model_label", "reward_score", "reward_delta_tuned_minus_orig", "embedding512_l2_delta"):
        _plot_scatter_matplotlib(
            comb_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"exp3_umap8192_color_{color_col}.png",
            title=f"Exp3 8192 UMAP colored by {color_col}",
        )
        _plot_scatter_matplotlib(
            comb_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"exp3_umap512_color_{color_col}.png",
            title=f"Exp3 512 UMAP colored by {color_col}",
        )
        hover = ["seed", "model_label", "reward_score", "reward_delta_tuned_minus_orig", "embedding512_l2_delta"]
        _plot_with_plotly(
            comb_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"exp3_umap8192_color_{color_col}.html",
            title=f"Exp3 8192 UMAP colored by {color_col}",
            hover_cols=hover,
        )
        _plot_with_plotly(
            comb_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"exp3_umap512_color_{color_col}.html",
            title=f"Exp3 512 UMAP colored by {color_col}",
            hover_cols=hover,
        )

    paired_rewards = pair_reward.reset_index()
    plt.figure(figsize=(7, 7))
    plt.scatter(paired_rewards["orig"], paired_rewards["tuned"], s=26, alpha=0.85)
    lo = float(min(paired_rewards["orig"].min(), paired_rewards["tuned"].min()))
    hi = float(max(paired_rewards["orig"].max(), paired_rewards["tuned"].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    plt.xlabel("orig reward")
    plt.ylabel("tuned reward")
    plt.title("Exp3 reward scatter: orig vs tuned")
    plt.tight_layout()
    plt.savefig(out_dir / "exp3_reward_scatter_orig_vs_tuned.png", dpi=180)
    plt.close()

    summary = {
        "n_pairs": int(len(pair_reward)),
        "umap8192": meta8192,
        "umap512": meta512,
        "reward_delta": _pairwise_summary(comb_df, key="seed", value="reward_score"),
        "embedding512_l2_delta_mean": float(np.mean(list(pair_emb_l2.values()))),
        "embedding512_l2_delta_median": float(np.median(list(pair_emb_l2.values()))),
        "silhouette_model_8192": _safe_silhouette(comb8192, comb_df["model_label"]),
        "silhouette_model_512": _safe_silhouette(comb512, comb_df["model_label"]),
    }
    _write_json(out_dir / "exp3_summary.json", summary)
    return {
        "df": comb_df,
        "emb8192": comb8192,
        "emb512": comb512,
        "summary": summary,
    }


def _run_exp2b(
    reward_model,
    sigma_aug,
    cfg: AnalysisConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    seed_table = _make_seed_table(
        start_seed=cfg.exp2b_seed_start,
        n=cfg.exp2b_n_samples,
        truncation_psi=cfg.exp2b_truncation,
        regime_name=f"trunc_{cfg.exp2b_truncation:.2f}",
    )
    da_tuned, cond, pads = _load_generator_assets(cfg.tuned_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc, samples, shape = _prepare_sigma_sampling(da_tuned.G, pads_vals=pads, shape_res=cfg.shape_res)
    df, emb8192, emb512, z_codes = _collect_features_for_table(
        table=seed_table,
        generator_label="tuned",
        G=da_tuned.G,
        conditioning_params=cond,
        mudc=mudc,
        samples=samples,
        shape=shape,
        reward_model=reward_model,
        sigma_aug=sigma_aug,
        cfg=cfg,
        device=device,
        desc="exp2b/tuned_sigma_sampling",
    )
    base = _run_tail_analysis_from_table(
        df=df,
        emb8192=emb8192,
        emb512=emb512,
        z_codes=z_codes,
        tail_k=cfg.tail_k,
        out_dir=out_dir,
        prefix="exp2b",
    )
    base["summary"]["n_samples"] = int(len(df))
    base["summary"]["shape"] = tuple(int(v) for v in shape[1:4])
    _write_json(out_dir / "exp2b_summary.json", base["summary"])
    return base


def _build_identity_seed_table(start_seed: int, n: int, truncation_psi: float, regime_name: str) -> pd.DataFrame:
    return _make_seed_table(start_seed=start_seed, n=n, truncation_psi=truncation_psi, regime_name=regime_name)


def _render_identity_bank(
    network_pkl: str,
    seed_table: pd.DataFrame,
    cfg: AnalysisConfig,
    device: torch.device,
    generator_label: str,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    da, _, _ = _load_generator_assets(network_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    camera, conditioning = _get_single_img_cam_bundle(da)
    camera = camera.to(device)
    conditioning = conditioning.to(device)
    z_batch = torch.cat([_seed_to_z(int(seed), device)[0] for seed in seed_table["seed"].tolist()], dim=0)
    rgb = _render_rgb_batch(
        da=da,
        z_batch=z_batch,
        camera_params=camera,
        conditioning_params=conditioning,
        neural_rendering_resolution=cfg.rgb_metric_resolution,
        truncation_psi=cfg.identity_match_truncation,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=cfg.identity_render_batch_size,
        desc=f"identity_bank/{generator_label}/rgb",
    )
    face_model = _load_face_identity_model(device)
    embeddings = _compute_face_embeddings(
        face_model,
        rgb,
        batch_size=cfg.identity_batch_size,
        device=device,
        desc=f"identity_bank/{generator_label}/face_emb",
    )
    del face_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta = seed_table.copy().reset_index(drop=True)
    meta["generator_label"] = generator_label
    return meta, embeddings


def _match_identity_pairs(
    orig_meta: pd.DataFrame,
    orig_emb: torch.Tensor,
    tuned_meta: pd.DataFrame,
    tuned_emb: torch.Tensor,
    topk: int,
    target_pairs: int,
    min_cosine: float,
) -> pd.DataFrame:
    orig_np = F.normalize(orig_emb, dim=1).cpu().numpy().astype(np.float32)
    tuned_np = F.normalize(tuned_emb, dim=1).cpu().numpy().astype(np.float32)
    sim = orig_np @ tuned_np.T
    topk = min(topk, tuned_np.shape[0])
    candidate_rows: List[Dict[str, Any]] = []
    for i in range(sim.shape[0]):
        idx = np.argpartition(sim[i], -topk)[-topk:]
        idx = idx[np.argsort(sim[i, idx])[::-1]]
        for rank, j in enumerate(idx.tolist()):
            candidate_rows.append(
                {
                    "orig_idx": int(i),
                    "tuned_idx": int(j),
                    "identity_cosine": float(sim[i, j]),
                    "nn_rank": int(rank),
                    "seed_orig": int(orig_meta.iloc[i]["seed"]),
                    "seed_tuned": int(tuned_meta.iloc[j]["seed"]),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.loc[candidates["identity_cosine"] >= min_cosine].sort_values(
        ["identity_cosine", "nn_rank"], ascending=[False, True]
    )
    used_orig: set[int] = set()
    used_tuned: set[int] = set()
    chosen: List[Dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        if row.orig_idx in used_orig or row.tuned_idx in used_tuned:
            continue
        used_orig.add(int(row.orig_idx))
        used_tuned.add(int(row.tuned_idx))
        chosen.append(row._asdict())
        if len(chosen) >= target_pairs:
            break
    return pd.DataFrame(chosen)


def _run_identity_matched_exp(
    reward_model,
    sigma_aug,
    cfg: AnalysisConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tuned_bank_table = _build_identity_seed_table(
        start_seed=cfg.tuned_identity_bank_seed_start,
        n=cfg.tuned_identity_bank_size,
        truncation_psi=cfg.identity_match_truncation,
        regime_name="identity_bank_tuned",
    )
    orig_query_table = _build_identity_seed_table(
        start_seed=cfg.orig_identity_query_seed_start,
        n=cfg.orig_identity_query_size,
        truncation_psi=cfg.identity_match_truncation,
        regime_name="identity_query_orig",
    )
    tuned_meta, tuned_face_emb = _render_identity_bank(
        network_pkl=cfg.tuned_pkl,
        seed_table=tuned_bank_table,
        cfg=cfg,
        device=device,
        generator_label="tuned",
    )
    orig_meta, orig_face_emb = _render_identity_bank(
        network_pkl=cfg.orig_pkl,
        seed_table=orig_query_table,
        cfg=cfg,
        device=device,
        generator_label="orig",
    )
    matches = _match_identity_pairs(
        orig_meta=orig_meta,
        orig_emb=orig_face_emb,
        tuned_meta=tuned_meta,
        tuned_emb=tuned_face_emb,
        topk=cfg.identity_match_topk,
        target_pairs=cfg.identity_match_target_pairs,
        min_cosine=cfg.identity_match_min_cosine,
    )
    if len(matches) == 0:
        raise RuntimeError("No identity-matched pairs found; lower identity_match_min_cosine or increase bank size.")

    da_orig, cond_orig, pads_orig = _load_generator_assets(cfg.orig_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc_orig, samples_orig, shape_orig = _prepare_sigma_sampling(da_orig.G, pads_vals=pads_orig, shape_res=cfg.shape_res)
    da_tuned, cond_tuned, pads_tuned = _load_generator_assets(cfg.tuned_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc_tuned, samples_tuned, shape_tuned = _prepare_sigma_sampling(da_tuned.G, pads_vals=pads_tuned, shape_res=cfg.shape_res)

    orig_records = []
    tuned_records = []
    orig_8192_list = []
    tuned_8192_list = []
    orig_512_list = []
    tuned_512_list = []
    orig_z_list = []
    tuned_z_list = []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="identity_matched/sigma_reward"):
        orig_vol, orig_z = _sample_sigma_volume(
            mudc=mudc_orig,
            G=da_orig.G,
            conditioning_params=cond_orig,
            samples=samples_orig,
            shape=shape_orig,
            seed=int(row.seed_orig),
            truncation_psi=cfg.identity_match_truncation,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        tuned_vol, tuned_z = _sample_sigma_volume(
            mudc=mudc_tuned,
            G=da_tuned.G,
            conditioning_params=cond_tuned,
            samples=samples_tuned,
            shape=shape_tuned,
            seed=int(row.seed_tuned),
            truncation_psi=cfg.identity_match_truncation,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        orig8192, orig512, reward_orig = _reward_forward(reward_model, sigma_aug, orig_vol, device)
        tuned8192, tuned512, reward_tuned = _reward_forward(reward_model, sigma_aug, tuned_vol, device)

        sigma_l1 = float((tuned_vol - orig_vol).abs().mean().item())
        sigma_l2 = float(torch.sqrt((tuned_vol - orig_vol).square().mean()).item())
        latent_l2 = float(np.linalg.norm(tuned_z - orig_z))

        shared = {
            "pair_id": int(len(orig_records)),
            "seed_orig": int(row.seed_orig),
            "seed_tuned": int(row.seed_tuned),
            "identity_cosine": float(row.identity_cosine),
            "latent_l2_between_pair": latent_l2,
            "sigma_pair_l1": sigma_l1,
            "sigma_pair_l2": sigma_l2,
        }
        orig_records.append(
            {
                **shared,
                "seed": int(row.seed_orig),
                "generator_label": "orig",
                "model_label": "orig",
                "reward_score": reward_orig,
            }
        )
        tuned_records.append(
            {
                **shared,
                "seed": int(row.seed_tuned),
                "generator_label": "tuned",
                "model_label": "tuned",
                "reward_score": reward_tuned,
            }
        )
        orig_8192_list.append(orig8192)
        tuned_8192_list.append(tuned8192)
        orig_512_list.append(orig512)
        tuned_512_list.append(tuned512)
        orig_z_list.append(orig_z)
        tuned_z_list.append(tuned_z)

    orig_df = pd.DataFrame(orig_records)
    tuned_df = pd.DataFrame(tuned_records)
    comb_df = pd.concat([orig_df, tuned_df], ignore_index=True)
    comb8192 = np.concatenate([np.stack(orig_8192_list), np.stack(tuned_8192_list)], axis=0).astype(np.float32)
    comb512 = np.concatenate([np.stack(orig_512_list), np.stack(tuned_512_list)], axis=0).astype(np.float32)
    combz = np.concatenate([np.stack(orig_z_list), np.stack(tuned_z_list)], axis=0).astype(np.float32)

    umap8192, meta8192 = _compute_umap(comb8192, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    umap512, meta512 = _compute_umap(comb512, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.umap_random_state, cfg.pca_components)
    comb_df["umap8192_x"] = umap8192[:, 0]
    comb_df["umap8192_y"] = umap8192[:, 1]
    comb_df["umap512_x"] = umap512[:, 0]
    comb_df["umap512_y"] = umap512[:, 1]

    reward_delta_map = (
        comb_df.pivot(index="pair_id", columns="model_label", values="reward_score")["tuned"]
        - comb_df.pivot(index="pair_id", columns="model_label", values="reward_score")["orig"]
    ).to_dict()
    comb_df["reward_delta_tuned_minus_orig"] = comb_df["pair_id"].map(reward_delta_map)
    emb_l2_map = {
        int(i): float(np.linalg.norm(np.stack(tuned_512_list)[i] - np.stack(orig_512_list)[i]))
        for i in range(len(orig_512_list))
    }
    comb_df["embedding512_l2_delta"] = comb_df["pair_id"].map(emb_l2_map)

    matches.to_csv(out_dir / "identity_matched_pairs.csv", index=False)
    comb_df.to_csv(out_dir / "identity_matched_embedding_samples.csv", index=False)
    _save_npz(out_dir / "identity_matched_embeddings.npz", emb8192=comb8192, emb512=comb512, z_codes=combz)

    for color_col in ("model_label", "identity_cosine", "reward_delta_tuned_minus_orig", "sigma_pair_l1"):
        _plot_scatter_matplotlib(
            comb_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"identity_match_umap8192_color_{color_col}.png",
            title=f"Identity matched 8192 UMAP colored by {color_col}",
        )
        _plot_scatter_matplotlib(
            comb_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"identity_match_umap512_color_{color_col}.png",
            title=f"Identity matched 512 UMAP colored by {color_col}",
        )
        hover = [
            "pair_id",
            "seed_orig",
            "seed_tuned",
            "model_label",
            "identity_cosine",
            "reward_score",
            "reward_delta_tuned_minus_orig",
            "sigma_pair_l1",
        ]
        _plot_with_plotly(
            comb_df,
            x="umap8192_x",
            y="umap8192_y",
            color_col=color_col,
            out_path=out_dir / f"identity_match_umap8192_color_{color_col}.html",
            title=f"Identity matched 8192 UMAP colored by {color_col}",
            hover_cols=hover,
        )
        _plot_with_plotly(
            comb_df,
            x="umap512_x",
            y="umap512_y",
            color_col=color_col,
            out_path=out_dir / f"identity_match_umap512_color_{color_col}.html",
            title=f"Identity matched 512 UMAP colored by {color_col}",
            hover_cols=hover,
        )

    summary = {
        "n_tuned_bank": int(len(tuned_meta)),
        "n_orig_query": int(len(orig_meta)),
        "n_pairs": int(len(matches)),
        "identity_cosine_mean": float(matches["identity_cosine"].mean()),
        "identity_cosine_min": float(matches["identity_cosine"].min()),
        "reward_delta": _pairwise_summary(comb_df, key="pair_id", value="reward_score"),
        "sigma_pair_l1_mean": float(orig_df["sigma_pair_l1"].mean()),
        "sigma_pair_l2_mean": float(orig_df["sigma_pair_l2"].mean()),
        "latent_l2_between_pair_mean": float(orig_df["latent_l2_between_pair"].mean()),
        "embedding512_l2_delta_mean": float(np.mean(list(emb_l2_map.values()))),
        "umap8192": meta8192,
        "umap512": meta512,
        "silhouette_model_8192": _safe_silhouette(comb8192, comb_df["model_label"]),
        "silhouette_model_512": _safe_silhouette(comb512, comb_df["model_label"]),
    }
    _write_json(out_dir / "identity_matched_summary.json", summary)
    return {"df": comb_df, "matches": matches, "summary": summary}


def _refresh_fid_bundle(cfg: AnalysisConfig) -> None:
    out_dir = _ensure_dir(Path(cfg.fid_results_dir))
    rows = [
        {
            "label": "original_baseline",
            "checkpoint": cfg.orig_pkl,
            "metric": "fid50k_full",
            "value": 4.122093442022415,
        },
        {
            "label": "same_loop_no_reward",
            "checkpoint": cfg.noreward_pkl,
            "metric": "fid50k_full",
            "value": 5.342319872286591,
        },
        {
            "label": "reward_tuned_model",
            "checkpoint": cfg.tuned_pkl,
            "metric": "fid50k_full",
            "value": 6.65655308876182,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fid_comparison.csv", index=False)

    original = float(df.loc[df["label"] == "original_baseline", "value"].iloc[0])
    noreward = float(df.loc[df["label"] == "same_loop_no_reward", "value"].iloc[0])
    tuned = float(df.loc[df["label"] == "reward_tuned_model", "value"].iloc[0])
    summary = {
        "dataset": cfg.dataset_path,
        "mirror": False,
        "num_gpus": 1,
        "metric": "fid50k_full",
        "rows": rows,
        "delta_noreward_minus_original": noreward - original,
        "delta_tuned_minus_original": tuned - original,
        "delta_tuned_minus_noreward": tuned - noreward,
        "best_model_by_fid": "original_baseline",
    }
    _write_json(out_dir / "fid_comparison.json", summary)
    with open(out_dir / "fid_comparison_summary.txt", "w") as f:
        f.write("fid50k_full comparison\n")
        for row in rows:
            f.write(f"{row['label']},{row['value']:.15f}\n")
        f.write(f"delta_noreward_minus_original,{(noreward - original):.15f}\n")
        f.write(f"delta_tuned_minus_original,{(tuned - original):.15f}\n")
        f.write(f"delta_tuned_minus_noreward,{(tuned - noreward):.15f}\n")
        f.write("better_model_by_fid,original_baseline\n")


def _write_master_summary(
    cfg: AnalysisConfig,
    exp1: Dict[str, Any],
    exp2: Dict[str, Any],
    exp2b: Dict[str, Any],
    exp3: Dict[str, Any],
    identity_matched: Dict[str, Any],
    out_dir: Path,
) -> None:
    summary = {
        "config": asdict(cfg),
        "exp1": exp1["summary"],
        "exp2": exp2["summary"],
        "exp2b": exp2b["summary"],
        "exp3": exp3["summary"],
        "identity_matched": identity_matched["summary"],
    }
    _write_json(out_dir / "analysis_summary.json", summary)


def main() -> None:
    cfg = AnalysisConfig()
    device = torch.device(cfg.device)
    out_dir = _ensure_dir(Path(cfg.results_dir))
    _refresh_fid_bundle(cfg)

    reward_model, _run_config, sigma_aug = _load_reward_assets(cfg, device)

    exp1_dir = _ensure_dir(out_dir / "exp1_regime_umap")
    exp2_dir = _ensure_dir(out_dir / "exp2_reward_tails")
    exp2b_dir = _ensure_dir(out_dir / "exp2b_reward_tails_psi1")
    exp3_dir = _ensure_dir(out_dir / "exp3_orig_vs_tuned")
    exp4_dir = _ensure_dir(out_dir / "exp4_identity_matched_orig_vs_tuned")

    exp1 = _run_exp1(reward_model, sigma_aug, cfg, device, exp1_dir)
    exp2 = _run_exp2(exp1, cfg, exp2_dir)
    exp2b = _run_exp2b(reward_model, sigma_aug, cfg, device, exp2b_dir)
    exp3 = _run_exp3(exp1, reward_model, sigma_aug, cfg, device, exp3_dir)
    identity_matched = _run_identity_matched_exp(reward_model, sigma_aug, cfg, device, exp4_dir)

    _write_master_summary(cfg, exp1, exp2, exp2b, exp3, identity_matched, out_dir)
    print(json.dumps({"results_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

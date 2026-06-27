# created by AM 26122025 to analyse the per epoch shift in reward dist

# calling the function 'run_combined_epoch_analysis' does three things
# 1. assemble all rewward scores over 200 fixed seeds for each epoch as saved in rwds_df_tick*.csv
# 2. run ols regression on the slopes of the reward score per epoch as a fnction of initial rwd score. this is to see if the geometry imporvement is influenced by the intiiial sssscore. eg do lower quality mesh tend to improve more or less than higher quality mesh over successive iterations?
# 3. plot the delta of reward score for each seed over the epochs. this is a trajectory for each seed that shows if its reward score tends to iimprove or get worse. in addition, colour of each line is determined by the initial score, with lower quality scores being blue adn better quality being red.
# 4. plot the histogram of centered reward dist as separate histogram plot for each epoch.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import linregress
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap


def assemble_meanrwds_per_epoch(tdir: Path) -> pd.DataFrame:
    rwds_df_fnlist = list(tdir.glob("rwds_df_tick*.csv"))
    # Keep ONLY the per-tick reward CSVs (rwds_df_tick_<N>.csv). The glob also matches
    # siblings like rwds_df_tick_0_visualised_seeds.csv / *_combined.csv whose names end
    # in non-numeric text -> the tick regex below yields NaN -> astype(int) crashes.
    rwds_df_fnlist = [fn for fn in rwds_df_fnlist if fn.name.replace("rwds_df_tick_", "").replace(".csv", "").isdigit()]
    idx = pd.Index([p.as_posix() for p in rwds_df_fnlist])

    ticks = idx.str.extract(r"(\d+)(?=\.csv$)").iloc[:, 0].astype(int)
    order = np.argsort(ticks.values)
    sorted_paths = [rwds_df_fnlist[i] for i in order]

    rwds = {fn.name: pd.read_csv(fn, index_col=2)[["rwd_score"]] for fn in sorted_paths}
    rwds = {k: rwds[k][["rwd_score"]].rename(columns={"rwd_score": f"rwd_score_epoch_{k.split('_')[-1].replace('.csv', '')}"}) for k in rwds.keys()}
    allrwds = pd.concat(rwds.values(), axis=1)
    print(f"mean rwd per epoch over {allrwds.shape[0]} seeds:\n{allrwds.mean(0)}")
    return allrwds


def regress_rwdscoredelta_on_initialscore(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    t = np.arange(df.shape[1])
    initial = df.iloc[:, 0].to_numpy()

    slopes = []
    for _, row in df.iterrows():
        y = row.to_numpy()
        res = linregress(t, y)
        slopes.append(res.slope)
    slopes = np.array(slopes)

    X = sm.add_constant(initial)  # intercept + initial
    ols = sm.OLS(slopes, X).fit()
    print("OLS slope_vs_initial:", ols.params[1], "p:", ols.pvalues[1], "r^2:", ols.rsquared)
    return ols


def plot_rwdscoredelta_trajectory(df: pd.DataFrame, out_path: Path, title_extra: str = ""):
    init_scores = df.iloc[:, 0].to_numpy()
    norm = Normalize(vmin=init_scores.min(), vmax=init_scores.max())
    cmap = LinearSegmentedColormap.from_list("redblue", ["red", "blue"])

    kk = df - df.iloc[:, [0]].values  # normalize initial score to zero
    traj = [[(y, x) for x, y in enumerate(row)] for row in kk.to_numpy()]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))

    for color, tk in zip(cmap(norm(init_scores)), traj):
        tk = np.array(tk)
        ax.plot(tk[:, 1], tk[:, 0], color=color, alpha=0.6)

    ax.set_xlabel("Epoch (t)")
    ax.set_ylabel("Delta reward from initial")
    ax.set_title(f"Trajectories colored by initial score {title_extra}")

    smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    fig.colorbar(smap, ax=ax, label="Initial score")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    return fig


def plot_centered_reward_hists(df, out_dir: Path, bins=40, dpi=300):
    """
    df: columns = rwd_score_epoch_*
    Saves reward_hist_centerzero_tick_{k}.jpg to out_dir
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        vals = df[col].to_numpy()
        centered = vals - vals.mean()

        plt.close("all")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(centered, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{col} (centered)")
        ax.set_xlabel("Reward (mean-centered)")
        ax.set_ylabel("Count")

        k = col.split("_")[-1]  # assumes col name format rwd_score_epoch_{k}
        out_path = out_dir / f"reward_hist_centerzero_tick_{k}.jpg"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


# Example usage:
# plot_centered_reward_hists(df, tdir)  # tdir is your run directory Path


def run_combined_epoch_analysis(rundir):
    rundir = Path(rundir)
    # Run
    df = assemble_meanrwds_per_epoch(rundir)

    ols = regress_rwdscoredelta_on_initialscore(df)

    plot_path = rundir / "rwd_delta_trajectories.jpg"
    _ = plot_rwdscoredelta_trajectory(df, plot_path, title_extra=f"(n={df.shape[0]} seeds)")

    # Save combined dataframe
    df.to_csv(rundir / "rwds_df_tick_combined.csv")

    # Save OLS summary
    with open(rundir / "rwd_initial_vs_slope_summary.txt", "w") as f:
        f.write(ols.summary().as_text())
        print(ols.summary())

    plot_centered_reward_hists(df=df, out_dir=rundir)


if __name__ == "__main__":
    rundir = Path("/media/krillman/240GB_DATA/training_runs_2/01427-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/")
    run_combined_epoch_analysis(rundir)

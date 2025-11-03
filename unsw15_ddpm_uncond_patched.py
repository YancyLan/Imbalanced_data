
import argparse, math, os, glob, sys, gc
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

# ----------------------
# Utils: DDPM components
# ----------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim*2)
        self.lin2 = nn.Linear(dim*2, dim)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / max(half, 1)
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        emb = F.silu(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class EpsModel(nn.Module):
    def __init__(self, xdim, tdim=128, hidden=1024):
        super().__init__()
        self.temb = TimeEmbedding(tdim)
        self.net = nn.Sequential(
            nn.Linear(xdim + tdim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, xdim),
        )
        self.tdim = tdim

    def forward(self, x, t):
        te = self.temb(t % self.tdim)
        h = torch.cat([x, te], dim=1)
        return self.net(h)

class DDPM:
    def __init__(self, xdim, T=500, beta_start=1e-4, beta_end=0.02, device="cpu", hidden=1024, lr=2e-4, amp=True):
        self.T = T
        self.device = device
        self.amp = amp
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)  # [T]
        self.betas = betas
        self.model = EpsModel(xdim, hidden=hidden).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.startswith("cuda") or device.startswith("mps")))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alphas_bar[t].reshape(-1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise, noise

    def train_epoch(self, x, batch_size=2048):
        self.model.train()
        n = x.shape[0]
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = x[idx].to(self.device, non_blocking=True)
            t = torch.randint(0, self.T, (xb.size(0),), device=self.device)
            with torch.cuda.amp.autocast(enabled=self.amp and (self.device.startswith("cuda") or self.device.startswith("mps"))):
                xt, noise = self.q_sample(xb, t)
                pred = self.model(xt, t)
                loss = F.mse_loss(pred, noise)
            self.opt.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                self.opt.step()
            total += loss.item() * xb.size(0)
        return total / n

    @torch.no_grad()
    def sample(self, n, xdim, batch=4096):
        self.model.eval()
        out = []
        for start in range(0, n, batch):
            bs = min(batch, n - start)
            x = torch.randn(bs, xdim, device=self.device)
            for t in range(self.T-1, -1, -1):
                tb = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
                eps = self.model(x, tb)
                a_t = 1 - self.betas[t]
                a_bar_t = self.alphas_bar[t]
                x = (1/torch.sqrt(a_t))*(x - (self.betas[t]/torch.sqrt(1-a_bar_t))*eps)
                if t > 0:
                    x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)
            out.append(x.cpu())
        return torch.cat(out, dim=0)

# ----------------------
# Data loading helpers
# ----------------------
def resolve_paths(csv_arg: str):
    p = Path(csv_arg)
    paths = []
    if p.is_dir():
        paths = sorted(list(Path(csv_arg).glob("*.csv")))
    else:
        # allow glob pattern or comma-separated list
        if any(ch in csv_arg for ch in ["*", "?", "["]):
            paths = [Path(x) for x in glob.glob(csv_arg)]
        elif "," in csv_arg:
            paths = [Path(x.strip()) for x in csv_arg.split(",") if x.strip()]
        else:
            paths = [p]
    return [str(x) for x in paths if Path(x).exists()]

def load_unsw(csv_arg: str, frac: float = 1.0, seed: int = 42):
    paths = resolve_paths(csv_arg)
    if len(paths) == 0:
        raise FileNotFoundError(f"No CSV files found from: {csv_arg}")
    frames = []
    for fp in paths:
        df = pd.read_csv(fp, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, axis=0, ignore_index=True)
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal")
        if "label" in df.columns:
            df.loc[df["label"]==0, "attack_cat"] = df.loc[df["label"]==0, "attack_cat"].replace("", "Normal")
    # stratified downsample by attack_cat if frac<1
    if 0 < frac < 1.0 and "attack_cat" in df.columns:
        df = (df.groupby("attack_cat", group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=seed))).reset_index(drop=True)
    elif 0 < frac < 1.0:
        df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    return df

# ----------------------
# Evaluation helpers
# ----------------------
def per_class_coverage_zspace(sub_df, num_cols, Xz, synth_z, outdir, scaler,
                              max_eval_per_class=3000, n_jobs=-1, seed=0):
    rng = np.random.RandomState(seed)
    # Build evaluation subset per class to keep NN tractable
    eval_indices = []
    if "attack_cat" in sub_df.columns:
        for cls, g in sub_df.groupby("attack_cat"):
            idx = g.index.values
            if len(idx) > max_eval_per_class:
                idx = rng.choice(idx, size=max_eval_per_class, replace=False)
            eval_indices.extend(idx.tolist())
    else:
        eval_indices = sub_df.index.values.tolist()
        if len(eval_indices) > max_eval_per_class:
            eval_indices = rng.choice(eval_indices, size=max_eval_per_class, replace=False).tolist()

    eval_indices = sorted(set(eval_indices))
    X_eval = Xz[eval_indices, :]
    y_eval = sub_df.loc[eval_indices, "attack_cat"].values if "attack_cat" in sub_df.columns else np.array(["ALL"]*len(eval_indices))

    # 真实->合成（在z空间）
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs, algorithm="auto")
    nbrs_synth.fit(synth_z)
    d_rs, _ = nbrs_synth.kneighbors(X_eval)  # distances to synthetic
    d_rs = d_rs[:,0]

    # 真实->真实(留一)（在z空间）
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs, algorithm="auto")
    nbrs_real.fit(X_eval)
    d_rr_all, nn_idx = nbrs_real.kneighbors(X_eval)
    d_rr = d_rr_all[:,1]  # second neighbor = leave-one-out

    rows = []
    for cls in np.unique(y_eval):
        mask = (y_eval == cls)
        rows.append({
            "class": cls,
            "n_eval": int(mask.sum()),
            "median_real2real": float(np.median(d_rr[mask])) if mask.any() else float("nan"),
            "median_real2synth": float(np.median(d_rs[mask])) if mask.any() else float("nan"),
            "relative_coverage": float(np.median(d_rr[mask]) / max(np.median(d_rs[mask]), 1e-12)) if mask.any() else float("nan"),
            "zspace_nn_median": float(np.median(d_rs[mask])) if mask.any() else float("nan"),
            "zspace_nn_p90": float(np.percentile(d_rs[mask], 90)) if mask.any() else float("nan"),
        })
    df_out = pd.DataFrame(rows).sort_values("relative_coverage", ascending=True)
    df_out.to_csv(Path(outdir)/"relative_coverage.csv", index=False)
    # For compatibility with previous naming
    df_out[["class","n_eval","zspace_nn_median","zspace_nn_p90"]].to_csv(Path(outdir)/"per_class_coverage_zspace.csv", index=False)
    return df_out

def distribution_shift_summary(sub_df, num_cols, Xz, synth_z, outdir, seed=0):
    # Train a light classifier on REAL (z-space), predict on SYNTH (z-space)
    if "attack_cat" not in sub_df.columns:
        return None
    y = sub_df["attack_cat"].values
    clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1, class_weight=None, max_depth=None)
    clf.fit(Xz, y)
    y_pred_synth = clf.predict(synth_z)

    real_counts = pd.Series(y).value_counts().sort_index()
    synth_counts = pd.Series(y_pred_synth).value_counts().reindex(real_counts.index).fillna(0).astype(int)
    dist_df = pd.DataFrame({
        "class": real_counts.index,
        "real_count": real_counts.values,
        "real_share": (real_counts/real_counts.sum()).values.round(6),
        "synth_predicted_count": synth_counts.values,
        "synth_share": (synth_counts/synth_counts.sum()).values.round(6),
        "synth/real_ratio": (synth_counts/real_counts).replace([np.inf, -np.inf], np.nan).fillna(0).round(6)
    })
    dist_df = dist_df.sort_values("synth/real_ratio")
    dist_df.to_csv(Path(outdir)/"distribution_shift_summary.csv", index=False)
    return dist_df

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to CSV or directory or glob pattern. Supports multiple files via comma separation.")
    ap.add_argument("--frac", type=float, default=1.0, help="Fraction of rows to use (keep imbalance). Default: 1.0 (use all).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500, help="diffusion steps T")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on GPU).")
    ap.add_argument("--outdir", type=str, default="out_uncond")
    ap.add_argument("--max_eval_per_class", type=int, default=3000, help="Cap eval items per class for NN metrics.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[INFO] Using device: {device}  AMP: {args.amp}")

    # 1) Load data
    df = load_unsw(args.csv, frac=args.frac, seed=args.seed)
    print(f"[INFO] Loaded shape: {df.shape}")

    # 2) Select numeric features; drop id/label-like columns
    drop_cols = [c for c in ["label", "id"] if c in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in drop_cols]
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for unconditional model.")
    X_num = df[num_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # 3) Standardize to z-space
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_num.values).astype(np.float32)

    # 4) Train unconditional DDPM on z-space
    ddpm = DDPM(xdim=Xz.shape[1], T=args.steps, device=device, hidden=args.hidden, lr=args.lr, amp=args.amp)
    Xtorch = torch.from_numpy(Xz)

    for ep in range(1, args.epochs+1):
        loss = ddpm.train_epoch(Xtorch, batch_size=args.batch_size)
        print(f"[Epoch {ep}/{args.epochs}] loss={loss:.6f}")
        sys.stdout.flush()

    # 5) Sample synthetic in z-space (same size as training set)
    synth_z = ddpm.sample(n=Xz.shape[0], xdim=Xz.shape[1], batch=max(1024, args.batch_size)).cpu().numpy()

    # 6) Evaluations

    # 6a) Per-class coverage in z-space + Relative coverage (real->real vs real->synth)
    rel_cov = per_class_coverage_zspace(
        sub_df=df, num_cols=num_cols, Xz=Xz, synth_z=synth_z, outdir=args.outdir, scaler=scaler,
        max_eval_per_class=args.max_eval_per_class, n_jobs=-1, seed=args.seed
    )
    print("\n[Per-class coverage in z-space & Relative coverage (smaller is worse):]")
    print(rel_cov.head(20))

    # 6b) Distribution shift summary (classifier view)
    dist_df = distribution_shift_summary(df, num_cols, Xz, synth_z, args.outdir, seed=args.seed)
    if dist_df is not None:
        print("\n[Distribution shift summary (synth/real_ratio):]")
        print(dist_df.head(20))

    # 7) Save artifacts
    # Save a small sample of rows to keep disk light
    n_preview = min(10000, Xz.shape[0])
    pd.DataFrame(Xz[:n_preview], columns=num_cols).to_csv(Path(args.outdir)/"train_subset_numeric_zspace_preview.csv", index=False)
    pd.DataFrame(synth_z[:n_preview], columns=num_cols).to_csv(Path(args.outdir)/"synthetic_numeric_zspace_preview.csv", index=False)

    # Also save scalers for later conditional runs
    np.save(Path(args.outdir)/"scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(Path(args.outdir)/"scaler_scale.npy", scaler.scale_.astype(np.float32))

    print(f"\n[DONE] Artifacts saved in: {args.outdir}")

if __name__ == "__main__":
    main()

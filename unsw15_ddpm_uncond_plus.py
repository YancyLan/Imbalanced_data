
import argparse, math, os, glob, sys, gc, copy
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

# ----------------------
# Schedules
# ----------------------
def cosine_beta_schedule(T, s: float = 0.008):
    # from Nichol & Dhariwal 2021
    steps = T + 1
    t = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)

# ----------------------
# EMA helper
# ----------------------
class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self.decay
        msd = model.state_dict()
        for name, p in self.shadow.state_dict().items():
            p.copy_(p * d + msd[name] * (1.0 - d))

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
            -math.log(10000) * torch.arange(0, max(half,1), device=t.device).float() / max(half,1)
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
    def __init__(self, xdim, T=500, device="cpu", hidden=1024, lr=2e-4, amp=True,
                 schedule="cosine", ema_decay=0.999, use_ema=True):
        self.T = T
        self.device = device
        self.amp = amp
        # schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(T).to(device)
        else:
            betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)  # [T]
        # model & opt
        self.model = EpsModel(xdim, hidden=hidden).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.startswith("cuda") or device.startswith("mps")))
        # EMA
        self.ema = EMAHelper(self.model, decay=ema_decay) if use_ema else None

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alphas_bar[t].reshape(-1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise, noise

    def train_epoch(self, x, batch_size=2048, grad_clip=1.0):
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
                pred = self.model(xt, t)            # eps-prediction
                loss = F.mse_loss(pred, noise)
            self.opt.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if grad_clip is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.opt.step()
            if self.ema is not None:
                self.ema.update(self.model)
            total += loss.item() * xb.size(0)
        return total / n

    @torch.no_grad()
    def sample(self, n, xdim, batch=4096, use_ema=True):
        model = self.ema.shadow if (use_ema and self.ema is not None) else self.model
        model.eval()
        out = []
        for start in range(0, n, batch):
            bs = min(batch, n - start)
            x = torch.randn(bs, xdim, device=self.device)
            for t in range(self.T-1, -1, -1):
                tb = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
                eps = model(x, tb)
                a_t = 1 - self.betas[t]
                a_bar_t = self.alphas_bar[t]
                x = (1/torch.sqrt(a_t))*(x - (self.betas[t]/torch.sqrt(1-a_bar_t))*eps)
                if t > 0:
                    x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)
            out.append(x.cpu())
        return torch.cat(out, dim=0)

# ----------------------
# Data helpers
# ----------------------
def resolve_paths(csv_arg: str):
    p = Path(csv_arg)
    paths = []
    if p.is_dir():
        paths = sorted(list(Path(csv_arg).glob("*.csv")))
    else:
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
    frames = [pd.read_csv(fp, low_memory=False) for fp in paths]
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

def detect_integer_columns(df_num: pd.DataFrame, max_unique_ratio: float = 0.9):
    int_cols = []
    for c in df_num.columns:
        s = df_num[c].dropna()
        if pd.api.types.is_integer_dtype(s):
            int_cols.append(c); continue
        # heuristic: values are near integers AND not almost-all-unique continuous
        if len(s) == 0: 
            continue
        if np.allclose(s.values, np.round(s.values), atol=1e-6):
            int_cols.append(c); continue
        # if small integer-like range (e.g., ports 0-65535 not good), skip ratio check
    return int_cols

# ----------------------
# Evaluation helpers
# ----------------------
def per_class_coverage_zspace(sub_df, Xz, synth_z, outdir,
                              max_eval_per_class=3000, n_jobs=-1, seed=0):
    rng = np.random.RandomState(seed)
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

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs)
    nbrs_synth.fit(synth_z)
    d_rs, _ = nbrs_synth.kneighbors(X_eval)  # real->synth distances
    d_rs = d_rs[:,0]

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs)
    nbrs_real.fit(X_eval)
    d_rr_all, _ = nbrs_real.kneighbors(X_eval)
    d_rr = d_rr_all[:,1]  # leave-one-out

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
    df_out[["class","n_eval","zspace_nn_median","zspace_nn_p90"]].to_csv(Path(outdir)/"per_class_coverage_zspace.csv", index=False)
    return df_out

def distribution_shift_original_space(sub_df, transformer, synth_z, outdir, int_cols, seed=0):
    if "attack_cat" not in sub_df.columns:
        return None
    # Real in original space (no dequant noise)
    num_cols = sub_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["label", "id"]]
    X_real = sub_df[num_cols].copy()
    # Synth in original space: inverse transform from z
    synth_orig = transformer.inverse_transform(synth_z)
    X_synth = pd.DataFrame(synth_orig, columns=num_cols)

    # Round & clip integer columns for both
    for c in int_cols:
        if c in X_real.columns:
            # clip synth to real min/max before rounding to avoid out-of-range
            mn, mx = X_real[c].min(), X_real[c].max()
            X_synth[c] = np.clip(X_synth[c].values, mn, mx)
            X_synth[c] = np.rint(X_synth[c].values).astype(X_real[c].dtype if pd.api.types.is_integer_dtype(X_real[c]) else np.int64)
            X_real[c] = np.rint(X_real[c].values).astype(X_real[c].dtype if pd.api.types.is_integer_dtype(X_real[c]) else np.int64)

    # Train RF on REAL-original, predict on SYNTH-original
    y = sub_df["attack_cat"].values
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1, class_weight=None, max_depth=None)
    clf.fit(X_real.values, y)
    y_pred_synth = clf.predict(X_synth.values)

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
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV/dir/glob/comma-separated list")
    ap.add_argument("--frac", type=float, default=1.0, help="Fraction of rows (stratified by attack_cat if available)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=1000, help="diffusion steps T")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    ap.add_argument("--outdir", type=str, default="out_uncond_plus")
    ap.add_argument("--max_eval_per_class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transform", type=str, default="quantile", choices=["zscore","quantile","power"],
                    help="Feature transform before training (default: quantile rank-gauss)")
    ap.add_argument("--dequantize_int", action="store_true", help="Add U(-0.5,0.5) noise to integer-like columns during training")
    ap.add_argument("--schedule", type=str, default="cosine", choices=["cosine","linear"])
    ap.add_argument("--ema", action="store_true", help="Use EMA weights for sampling (default on in this script)")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    args = ap.parse_args()

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
    print(f"[INFO] Using device: {device} AMP={args.amp}  schedule={args.schedule}  transform={args.transform}  dequant={args.dequantize_int}")

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load
    df = load_unsw(args.csv, frac=args.frac, seed=args.seed)
    print(f"[INFO] Loaded: {df.shape}")

    # 2) Numeric features
    drop_cols = [c for c in ["label", "id"] if c in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in drop_cols]
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found.")
    X_num = df[num_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # 3) Detect integer-like columns (before any transform)
    int_cols = detect_integer_columns(X_num)
    print(f"[INFO] Detected integer-like columns: {len(int_cols)}  e.g., {int_cols[:8]}")

    # Keep a clean copy for original-space evaluation (no dequant noise)
    X_orig_for_eval = X_num.copy()

    # 4) Optional dequantization noise for training
    rng = np.random.default_rng(args.seed)
    if args.dequantize_int and len(int_cols) > 0:
        noise = rng.uniform(-0.5, 0.5, size=(X_num.shape[0], len(int_cols))).astype(np.float32)
        X_num.loc[:, int_cols] = (X_num[int_cols].values + noise)

    # 5) Transform -> z-space
    if args.transform == "zscore":
        transformer = StandardScaler()
    elif args.transform == "quantile":
        transformer = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, X_num.shape[0]))
    else:  # power
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    Xz = transformer.fit_transform(X_num.values).astype(np.float32)

    # 6) Train DDPM
    ddpm = DDPM(xdim=Xz.shape[1], T=args.steps, device=device, hidden=args.hidden, lr=args.lr,
                amp=args.amp, schedule=args.schedule, ema_decay=args.ema_decay, use_ema=True)
    Xtorch = torch.from_numpy(Xz)
    for ep in range(1, args.epochs+1):
        loss = ddpm.train_epoch(Xtorch, batch_size=args.batch_size, grad_clip=1.0)
        print(f"[Epoch {ep}/{args.epochs}] loss={loss:.6f}")
        sys.stdout.flush()

    # 7) Sample (z-space) using EMA
    synth_z = ddpm.sample(n=Xz.shape[0], xdim=Xz.shape[1], batch=max(1024, args.batch_size), use_ema=True).cpu().numpy()

    # 8) Coverage in z-space (relative coverage)
    rel_cov = per_class_coverage_zspace(df, Xz, synth_z, args.outdir, max_eval_per_class=args.max_eval_per_class, n_jobs=-1, seed=args.seed)
    print("\n[Relative coverage in z-space (smaller=worse)]:")
    print(rel_cov.head(20))

    # 9) Distribution shift in ORIGINAL space with rounding for integer columns
    dist_df = distribution_shift_original_space(df, transformer, synth_z, args.outdir, int_cols, seed=args.seed)
    if dist_df is not None:
        print("\n[Distribution shift (original space)]:")
        print(dist_df.head(20))

    # 10) Save small previews and transformer params
    n_preview = min(10000, Xz.shape[0])
    pd.DataFrame(Xz[:n_preview], columns=num_cols).to_csv(Path(args.outdir)/"train_subset_numeric_zspace_preview.csv", index=False)
    pd.DataFrame(synth_z[:n_preview], columns=num_cols).to_csv(Path(args.outdir)/"synthetic_numeric_zspace_preview.csv", index=False)

    # save transformer type
    with open(Path(args.outdir)/"transform_info.txt", "w", encoding="utf-8") as f:
        f.write(f"transform={args.transform}\n")
        f.write(f"dequantize_int={args.dequantize_int}\n")
        f.write(f"int_cols={','.join(int_cols[:])}\n")

    print(f"\n[DONE] Artifacts in: {args.outdir}")

if __name__ == "__main__":
    main()

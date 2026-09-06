"""Frozen-backbone feature extraction + caching (Day 1 Block 3, Kaggle GPU).

Contract (CONTRACTS.md §3): ``data/cache/{split}_{transform}_{param}.npy``
+ ``index.csv``, one row per image in manifest order. CLI takes
``--split --transform --param``; one invocation writes one cache pair.

Pipeline per row: PIL open -> ``transforms.apply`` (skipped for ``clean``)
-> CLIP preprocessing (224 center-crop + CLIP mean/std, pinned in
``configs/transforms.yaml``) -> frozen ``open_clip`` ViT-L/14 (fp16 on CUDA)
-> L2-normalized embedding (float32 on disk).

``torch`` / ``open_clip`` / ``torchvision`` are lazy-imported so ``--help``
and the pure-helper unit tests run on CPU-only hosts without them.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_PREPROC = REPO_ROOT / "configs" / "transforms.yaml"

MODEL_DEFAULT = "ViT-L-14"
PRETRAINED_DEFAULT = "openai"


def parse_param(transform: str, raw: str | None):
    """Parse a CLI ``--param`` string into the registry value.

    ``null``/``none``/empty -> None (clean); ints stay ints (70, 20, 80),
    floats stay floats (0.5, 0.02); anything else (``h``) stays a string.
    """
    if raw is None or raw.strip().lower() in ("", "null", "none"):
        return None
    _ = transform  # value parsing is transform-agnostic today
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def load_preprocessing_cfg(path: Path = DEFAULT_PREPROC) -> dict:
    """Load the ``preprocessing`` block (size/mean/std) from transforms.yaml."""
    import yaml

    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["preprocessing"]
    assert isinstance(cfg["size"], int) and len(cfg["mean"]) == 3
    assert len(cfg["std"]) == 3 and cfg["center_crop"] is True
    return cfg


def build_preprocess(cfg: dict):
    """torchvision CLIP preprocessing: Resize -> CenterCrop -> Norm."""
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    size = int(cfg["size"])
    return T.Compose([
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=list(cfg["mean"]), std=list(cfg["std"])),
    ])


def load_backbone(model_name: str, pretrained: str, device: str, precision: str):
    """Frozen open_clip model -> (model, embed_dim)."""
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device.startswith("cuda") and precision == "fp16":
        model.half()
    dim = getattr(getattr(model, "visual", None), "output_dim", None)
    if dim is None:  # fallback: probe with a dummy batch
        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224, device=device)
            if device.startswith("cuda") and precision == "fp16":
                probe = probe.half()
            dim = int(model.encode_image(probe).shape[1])
    return model, int(dim)


def read_split_rows(manifest: Path, split: str, limit: int = 0) -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"no rows for split={split!r} in {manifest}")
    return rows


def write_index(rows: list[dict], path: Path, extra: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "label", "source", "generator", "split", *extra]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})


def sample_aug_views(k: int, seed: int) -> list[dict]:
    """K deterministic (transform/chain, param) specs for augmentation-as-training.

    Policy: 15% chain (uniform over CHAINS), else uniform over the 15 single
    (name, param) pairs in SUPPORTED (clean excluded). Seeded -> reproducible.
    """
    import random

    from aigc_detect.transforms import CHAINS, SUPPORTED

    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    singles = [(t, p) for t, ps in SUPPORTED.items() if t != "clean" for p in ps]
    chains = sorted(CHAINS)
    if not singles:
        raise RuntimeError("empty transform table")
    rng = random.Random(seed)
    views = []
    for _ in range(k):
        if chains and rng.random() < 0.15:
            views.append({"transform": rng.choice(chains), "param": "chain",
                          "chain": True})
        else:
            t, p = singles[rng.randrange(len(singles))]
            views.append({"transform": t, "param": p, "chain": False})
    return views


def cache_stem(cache_dir: Path, split: str, transform: str, param) -> Path:
    from aigc_detect.features import cache_path

    return cache_dir / cache_path(cache_dir, split, transform, param).name


def extract(args: argparse.Namespace) -> dict:
    import numpy as np

    t_all = time.perf_counter()
    use_chain = args.chain is not None
    randaug = args.random_aug > 0
    if randaug and (use_chain or args.transform != "clean" or args.param is not None):
        raise ValueError("--random-aug cannot be combined with --transform/--param/--chain")
    if use_chain and args.transform != "clean":
        raise ValueError("--chain cannot be combined with --transform")
    name = args.chain if use_chain else args.transform
    param = "chain" if use_chain else parse_param(args.transform, args.param)
    rows = read_split_rows(Path(args.manifest), args.split, args.limit)
    views: list[dict] | None = None
    if randaug:
        views = sample_aug_views(args.random_aug, args.aug_seed)
        name, param = f"randaug{args.random_aug}", f"seed{args.aug_seed}"
    print(f"[extract] {len(rows)} rows split={args.split} "
          f"{'chain' if use_chain else 'transform'}={name} param={param}", flush=True)

    if not randaug and not use_chain and args.transform == "clean" and param is not None:
        raise ValueError(f"clean takes no param, got {param!r}")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from aigc_detect.transforms import apply, apply_chain

    torch.manual_seed(args.seed)
    preproc = build_preprocess(load_preprocessing_cfg(Path(args.preproc)))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = args.precision
    if precision == "auto":
        precision = "fp16" if device.startswith("cuda") else "fp32"
    print(f"[extract] backbone {args.model}/{args.pretrained} "
          f"device={device} precision={precision}", flush=True)

    class _Ds(Dataset):
        def __init__(self) -> None:
            self.broken: list[str] = []
            # randaug: one job per (row, view); rep-major order.
            self.jobs = ([(i, v) for v in (views or []) for i in range(len(rows))]
                         if randaug else [(i, None) for i in range(len(rows))])

        def __len__(self) -> int:
            return len(self.jobs)

        def __getitem__(self, idx: int):
            i, spec = self.jobs[idx]
            try:
                with Image.open(REPO_ROOT / rows[i]["image_path"]) as im:
                    img = im.convert("RGB")
                    if randaug:
                        assert spec is not None
                        if spec["chain"]:
                            img = apply_chain(img, spec["transform"])
                        else:
                            img = apply(img, spec["transform"], spec["param"])
                    elif use_chain:
                        img = apply_chain(img, args.chain)
                    elif args.transform != "clean":
                        img = apply(img, args.transform, param)
                    return preproc(img)
            except Exception as e:
                if not args.skip_broken:
                    raise RuntimeError(f"broken image: {rows[i]['image_path']}") from e
                self.broken.append(f"{rows[i]['image_path']}: {e}")
                return torch.zeros(3, 224, 224)

    ds = _Ds()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=device.startswith("cuda"))
    model, dim = load_backbone(args.model, args.pretrained, device, precision)
    total_imgs = len(ds)
    feats = np.empty((total_imgs, dim), dtype=np.float32)
    use_amp = device.startswith("cuda") and precision == "fp16"
    state = {"done": 0, "last_batch": time.perf_counter(), "stop": False}

    def _heartbeat() -> None:
        # Fires while the main loop blocks inside slow CPU transforms, so a
        # quiet log means "working", not "stuck". Daemon: dies with the job.
        while not state["stop"]:
            time.sleep(args.heartbeat)
            if state["stop"]:
                break
            el = time.perf_counter() - t_all
            idle = time.perf_counter() - state["last_batch"]
            print(f"[extract] alive: {state['done']}/{total_imgs} "
                  f"({state['done'] / el:.1f} img/s avg, last batch {idle:.0f}s ago)",
                  flush=True)

    hb = None
    if args.heartbeat > 0:
        import threading

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
    done, t_fwd, next_report = 0, 0.0, args.progress_every
    with torch.no_grad():
        for batch in loader:
            t0 = time.perf_counter()
            batch = batch.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model.encode_image(batch).float()
            else:
                out = model.encode_image(batch).float()
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            t_fwd += time.perf_counter() - t0
            n = out.shape[0]
            feats[done:done + n] = out.cpu().numpy()
            done += n
            state.update(done=done, last_batch=time.perf_counter())
            if done >= next_report or done == total_imgs:
                el = time.perf_counter() - t_all
                eta = (total_imgs - done) / (done / el) if done else 0
                print(f"[extract] {done}/{total_imgs} ({done / el:.1f} img/s, "
                      f"eta {eta:.0f}s)", flush=True)
                next_report += args.progress_every
    state["stop"] = True

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_stem(cache_dir, args.split, name, param)
    npy_path = stem.with_suffix(".npy")
    np.save(npy_path, feats)
    if randaug:
        assert views is not None
        out_rows = [{**rows[i], "transform": spec["transform"],
                     "param": str(spec["param"])}
                    for spec in views for i in range(len(rows))]
        write_index(out_rows, stem.parent / (stem.name + ".index.csv"),
                    extra=("transform", "param"))
    else:
        write_index(rows, stem.parent / (stem.name + ".index.csv"))
    total = time.perf_counter() - t_all
    meta = {"model": args.model, "pretrained": args.pretrained, "device": device,
            "precision": precision, "batch_size": args.batch_size,
            "workers": args.workers, "split": args.split,
            "transform": name, "param": param, "chain": args.chain,
            "random_aug": args.random_aug, "aug_seed": args.aug_seed,
            "n_images": total_imgs,
            "broken": len(ds.broken), "seconds": round(total, 1),
            "imgs_per_sec": round(total_imgs / total, 1),
            "fwd_seconds": round(t_fwd, 1)}
    (stem.parent / (stem.name + ".meta.json")).write_text(json.dumps(meta, indent=2))
    if ds.broken:
        print(f"[extract] WARNING: {len(ds.broken)} broken, zero-filled "
              f"(first: {ds.broken[:3]})", flush=True)
    print(f"[extract] wrote {npy_path} {feats.shape} "
          f"{meta['imgs_per_sec']} img/s ({total:.0f}s total)", flush=True)
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Frozen CLIP extraction -> cache (CONTRACTS §3)")
    ap.add_argument("--split", required=True, help="manifest split (train/val/test/heldout)")
    ap.add_argument("--transform", default="clean",
                      help="e.g. clean, jpeg, blur (see transforms.yaml)")
    ap.add_argument("--param", default=None, help="e.g. 70; null/None for clean")
    ap.add_argument("--chain", default=None,
                      help="chain name (e.g. screenshot_repost); no --transform with it")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--preproc", type=Path, default=DEFAULT_PREPROC)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--pretrained", default=PRETRAINED_DEFAULT)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (+:0 ids ok)")
    ap.add_argument("--precision", default="auto", help="auto|fp16|fp32")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0=all; smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-broken", action="store_true", help="zero-fill unreadable images")
    ap.add_argument("--random-aug", type=int, default=0, metavar="K",
                      help="K random augmented views per row (overnight training set)")
    ap.add_argument("--aug-seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=1000,
                      help="progress line every N images")
    ap.add_argument("--heartbeat", type=int, default=30,
                      help="alive-line every N sec while blocked (0=off)")
    return ap


def main(argv: list[str] | None = None) -> None:
    extract(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

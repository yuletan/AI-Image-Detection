"""Day-1 manifest builder: WildFake subset + SID_Set stream -> manifest.csv.

Implements CONTRACTS.md section 1 (the builder lives in this module).
Schema: image_path,label,source,generator,split (see data/__init__.py).

Split plan (documented choice): heldout = ``ddim`` only (fake, never in
train). train/val/test are stratified over the remaining five generators
(celebahq, ffhq, sid_real, sid_synth, sid_tampered):

- train 20k: real 10k (4000 celebahq + 4000 ffhq + 2000 sid_real),
  fake 10k (5000 sid_synth + 5000 sid_tampered) -> balanced, 5 generators
- val 2k: 1000 real (400/400/200) + 1000 fake (500/500)
- test 4k: 2000 real (800/800/400) + 2000 fake (1000/1000)
- heldout 1k: 1000 ddim (fake-only unseen-generator probe)

Total 27k rows + header (~26k target + 1k heldout probe).
Dedupe by pHash before split; leak-check vs data/demo_benchmark/.
Deterministic: sorted scans + seeded shuffle (default seed 42).

Usage:
    PYTHONPATH=src python -m aigc_detect.data.manifest --out data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from collections import Counter
from pathlib import Path

from . import MANIFEST_COLUMNS, VALID_SPLITS, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
WILDFAKE_ROOT = REPO_ROOT / "data" / "raw" / "wildfake_subset" / "Images"
SID_DIR = REPO_ROOT / "data" / "raw" / "sid_set_subset"
DEMO_DIR = REPO_ROOT / "data" / "demo_benchmark"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "manifest.csv"

HELDOUT_GENERATOR = "ddim"
SPLIT_ORDER = ("train", "val", "test", "heldout")
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUOTAS: dict[str, dict[str, int]] = {
    "train": {"celebahq": 4000, "ffhq": 4000, "sid_real": 2000,
              "sid_synth": 5000, "sid_tampered": 5000},
    "val": {"celebahq": 400, "ffhq": 400, "sid_real": 200,
            "sid_synth": 500, "sid_tampered": 500},
    "test": {"celebahq": 800, "ffhq": 800, "sid_real": 400,
             "sid_synth": 1000, "sid_tampered": 1000},
    "heldout": {"ddim": 1000},
}

SID_GENERATORS = ("sid_real", "sid_synth", "sid_tampered")

# Status cadence inside tight loops (checked-in images). Small pools (tests)
# stay quiet; large pools report every _REPORT_EVERY files. Deterministic.
_REPORT_EVERY = 1000


def _log(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def _require_imagehash():
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:  # fail closed: never silently skip dedupe/leak check
        raise RuntimeError("ImageHash+Pillow required for pHash dedupe/leak check") from e


def phash_of(path: Path) -> str:
    """Perceptual hash (hex) of an image file."""
    _require_imagehash()
    import imagehash
    from PIL import Image

    with Image.open(path) as img:
        return str(imagehash.phash(img))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sid_label_and_generator(img_id: str, label: int) -> tuple[int, str]:
    """Map SID_Set example -> (manifest label, generator). 0 real; 1/2 fake."""
    label = int(label)
    if label == 0:
        return 0, "sid_real"
    if label == 1:
        return 1, "sid_synth"
    if label == 2:
        return 1, "sid_tampered"
    raise ValueError(f"unknown SID label {label!r} for {img_id!r}")


Pools = dict[str, list[tuple[Path, int]]]


def scan_wildfake(root: Path = WILDFAKE_ROOT, verbose: bool = True) -> Pools:
    """Scan Images/{Real,Difference-based}/* -> {generator: [(path, label)]}.

    Real/* -> label 0, Diffusion_based/* -> label 1; generator is the
    immediate child folder name, lowercased (celebahq/ffhq/ddim).
    Recursion-tolerant: files may sit in deeper subfolders (images/, ...).
    """
    t0 = time.perf_counter()
    pools: dict[str, list[tuple[Path, int]]] = {}
    for sub, label in (("Real", 0), ("Diffusion_based", 1)):
        basedir = root / sub
        if not basedir.is_dir():
            _log(f"[scan] {sub}/: missing, skipped", verbose=verbose)
            continue
        gendirs = sorted(p for p in basedir.iterdir() if p.is_dir())
        _log(f"[scan] {sub}/: {len(gendirs)} generator folders", verbose=verbose)
        for gendir in gendirs:
            gen = gendir.name.lower()
            _log(f"[scan] {sub}/{gendir.name} ...", verbose=verbose)
            files = sorted(
                p for p in gendir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
            )
            _log(f"[scan] {sub}/{gendir.name}: {len(files)} files", verbose=verbose)
            pools.setdefault(gen, []).extend((p, label) for p in files)
    total = sum(len(v) for v in pools.values())
    _log(f"[scan] done: {total} files in {time.perf_counter() - t0:.1f}s", verbose=verbose)
    return pools


def stream_sid(dest: Path, quotas: dict[str, int], seed: int = 42,
               max_examples: int = 60000, verbose: bool = True) -> Pools:
    """Stream saberzl/SID_Set, save needed images as JPEG under dest/.

    Returns {generator: [(path, label)]}. Reuses files already on disk
    ({generator}_{img_id}.jpg) without re-downloading. Label map: 0->real,
    1/2->fake. Quotas count post-dedupe candidates (top-up on phash dupes
    is handled by the caller via extra streaming headroom).
    """
    from datasets import load_dataset

    t0 = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[tuple[Path, int]]] = {g: [] for g in SID_GENERATORS}
    need = dict(quotas)
    # reuse pass first (deterministic: sorted filenames)
    for f in sorted(dest.glob("sid_*.jpg")):
        gen = "_".join(f.stem.split("_")[:2])
        if gen in need and need[gen] > 0 and len(pools[gen]) < quotas[gen]:
            lab = 0 if gen == "sid_real" else 1
            pools[gen].append((f, lab))
    for g in need:
        need[g] = quotas[g] - len(pools[g])
    reused = sum(len(v) for v in pools.values())
    _log(f"  sid reuse: {reused} on disk, still need {need}", verbose=verbose)
    if all(v <= 0 for v in need.values()):
        return pools
    _log(f"  sid streaming {sum(need.values())} images (this is network-bound) ...",
         verbose=verbose)
    ds = load_dataset("saberzl/SID_Set", streaming=True, split="train")
    have = {p for lst in pools.values() for p, _ in lst}
    streamed = saved = 0  # intake follows stream order (deterministic per revision)
    for ex in ds:
        if streamed >= max_examples:
            break
        if all(len(pools[g]) >= quotas[g] for g in quotas):
            break
        streamed += 1
        if verbose and streamed % 5000 == 0:
            got = {g: len(pools[g]) for g in quotas}
            _log(f"  sid: streamed={streamed} saved_new={saved} kept={got} "
                 f"({time.perf_counter() - t0:.0f}s)", verbose=verbose)
        try:
            lab, gen = sid_label_and_generator(ex["img_id"], ex["label"])
        except ValueError:
            continue
        if gen not in quotas or len(pools[gen]) >= quotas[gen]:
            continue
        safe = "".join(c for c in str(ex["img_id"]) if c.isalnum() or c in "-_") or "img"
        out = dest / f"{gen}_{safe}.jpg"
        if out in have:
            continue
        if not out.exists():
            img = ex["image"]
            if img is None:
                continue
            img.convert("RGB").save(out, format="JPEG", quality=90)
            saved += 1
        have.add(out)
        pools[gen].append((out, lab))
    short = {g: quotas[g] - len(pools[g]) for g in quotas if len(pools[g]) < quotas[g]}
    if short:
        raise RuntimeError(f"SID stream shortfall after {streamed} examples: {short}")
    _log(f"  sid done: streamed={streamed} saved_new={saved} ({time.perf_counter() - t0:.1f}s)",
         verbose=verbose)
    return pools


def select_split(pools: dict[str, list[tuple[Path, int]]],
                 quotas: dict[str, dict[str, int]], seed: int = 42,
                 rel_to: Path | None = None, verbose: bool = True) -> tuple[list[dict], int]:
    """pHash-dedupe each pool, then fill per-split quotas (seeded, deterministic).

    pools: {generator: [(path, label)]}. Phase 1 collects uniques per
    generator (skips count as true-dupe removals); phase 2 slices them
    across splits so no file is ever scanned twice. Raises RuntimeError
    on quota shortfall (fail closed, keep splits exact).
    """
    _require_imagehash()
    rng = random.Random(seed)
    need: dict[str, int] = {}
    for q in quotas.values():
        for gen, n in q.items():
            need[gen] = need.get(gen, 0) + n
    picked: dict[str, list[tuple[Path, int]]] = {}
    removed = 0
    seen: set[str] = set()
    for gen in sorted(need):
        pool = list(pools.get(gen, []))
        rng.shuffle(pool)
        _log(f"[dedupe] {gen}: need={need[gen]} pool={len(pool)}", verbose=verbose)
        t0 = time.perf_counter()
        sel: list[tuple[Path, int]] = []
        checked = 0
        for path, label in pool:
            if len(sel) >= need[gen]:
                break
            checked += 1
            h = phash_of(path)
            if h in seen:
                removed += 1
                continue
            seen.add(h)
            sel.append((path, label))
            if verbose and len(pool) > _REPORT_EVERY and checked % _REPORT_EVERY == 0:
                _log(f"[dedupe] {gen}: {checked}/{len(pool)} scanned, "
                     f"kept={len(sel)} dupes={removed} ({time.perf_counter() - t0:.0f}s)",
                     verbose=verbose)
        if len(sel) < need[gen]:
            raise RuntimeError(f"quota shortfall: gen={gen} got={len(sel)}/{need[gen]}")
        _log(f"[dedupe] {gen}: kept={len(sel)} ({time.perf_counter() - t0:.1f}s)",
             verbose=verbose)
        picked[gen] = sel
    rows: list[dict] = []
    for split in SPLIT_ORDER:
        group: list[dict] = []
        for gen, n in quotas.get(split, {}).items():
            chunk, picked[gen] = picked[gen][:n], picked[gen][n:]
            for path, label in chunk:
                p = Path(path)
                group.append({
                    "image_path": p.relative_to(rel_to).as_posix()
                    if rel_to else p.as_posix(),
                    "label": int(label),
                    "source": "sid_set" if gen in SID_GENERATORS else "wildfake",
                    "generator": gen,
                    "split": split,
                })
        rng.shuffle(group)
        rows.extend(group)
    return rows, removed


def check_demo_leak(rows: list[dict], demo_dir: Path = DEMO_DIR,
                    root: Path = REPO_ROOT, threshold: int = 5,
                    verbose: bool = True) -> dict:
    """Zero-overlap check vs quarantined demo set: exact sha256 OR pHash<=threshold.

    Returns {n_demo, n_checked, exact_hits, near_hits}. Raises on any hit.
    """
    _require_imagehash()
    import imagehash

    t0 = time.perf_counter()
    demo_files = sorted(
        p for p in demo_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES) if demo_dir.is_dir() else []
    _log(f"[leak] hashing {len(demo_files)} demo files ...", verbose=verbose)
    demo_exact = {sha256_of(p) for p in demo_files}
    demo_ph = [imagehash.hex_to_hash(phash_of(p)) for p in demo_files]
    _log(f"[leak] checking {len(rows)} rows (thr={threshold}) ...", verbose=verbose)
    exact = near = 0
    for i, r in enumerate(rows, 1):
        p = root / r["image_path"]
        if sha256_of(p) in demo_exact:
            exact += 1
            continue
        h = imagehash.hex_to_hash(phash_of(p))
        if any(h - d <= threshold for d in demo_ph):
            near += 1
        if verbose and len(rows) > _REPORT_EVERY and i % _REPORT_EVERY == 0:
            _log(f"[leak] {i}/{len(rows)} checked ({time.perf_counter() - t0:.0f}s)",
                 verbose=verbose)
    result = {"n_demo": len(demo_files), "n_checked": len(rows),
              "exact_hits": exact, "near_hits": near, "threshold": threshold}
    if exact or near:
        raise RuntimeError(f"DEMO LEAK: {result}")
    _log(f"[leak] clean: {len(rows)} rows vs {len(demo_files)} demo "
         f"({time.perf_counter() - t0:.1f}s)", verbose=verbose)
    return result


def validate_rows(rows: list[dict]) -> None:
    """Assert CONTRACTS schema: columns, labels {0,1}, splits in VALID_SPLITS."""
    for i, r in enumerate(rows):
        if [k for k in r] != MANIFEST_COLUMNS and set(r) != set(MANIFEST_COLUMNS):
            raise ValueError(f"row {i}: bad columns {sorted(r)}")
        if int(r["label"]) not in (0, 1):
            raise ValueError(f"row {i}: bad label {r['label']!r}")
        if r["split"] not in VALID_SPLITS:
            raise ValueError(f"row {i}: bad split {r['split']!r}")
        if not r["image_path"] or not r["generator"] or not r["source"]:
            raise ValueError(f"row {i}: empty field {r}")


def scale_quotas(quotas: dict[str, dict[str, int]], limit: int) -> dict[str, dict[str, int]]:
    total = sum(n for q in quotas.values() for n in q.values())
    if limit <= 0 or limit >= total:
        return quotas
    f = limit / total
    return {s: {g: max(1, round(n * f)) for g, n in q.items()} for s, q in quotas.items()}


def build_manifest(seed: int = 42, out: Path = DEFAULT_OUT, limit: int = 0,
                   sid_max: int = 60000, threshold: int = 5,
                   verbose: bool = True) -> dict:
    t_all = time.perf_counter()
    quotas = scale_quotas(QUOTAS, limit)
    need = Counter()
    for q in quotas.values():
        need.update(q)
    _log(f"[1/5] pools needed: {dict(need)}", verbose=verbose)
    pools = scan_wildfake(verbose=verbose)
    _log(f"wildfake pools: { {g: len(v) for g, v in pools.items()} }", verbose=verbose)
    sid_q = {g: need[g] for g in SID_GENERATORS if need.get(g, 0) > 0}
    if sid_q:
        # +~17% headroom so true pHash dupes can't cause a quota shortfall
        stream_q = {g: q + (q // 6) + 5 for g, q in sid_q.items()}
        _log(f"[2/5] streaming SID_Set for {sid_q} (fetch {stream_q}) ...", verbose=verbose)
        for g, lst in stream_sid(SID_DIR, stream_q, seed, sid_max,
                                verbose=verbose).items():
            pools.setdefault(g, []).extend(lst)
    else:
        _log("[2/5] SID skipped (no sid quota)", verbose=verbose)
    _log("[3/5] pHash dedupe + split ...", verbose=verbose)
    rows, removed = select_split(pools, quotas, seed, rel_to=REPO_ROOT, verbose=verbose)
    validate_rows(rows)
    _log(f"[4/5] validating {len(rows)} file paths ...", verbose=verbose)
    for i, r in enumerate(rows, 1):
        if not (REPO_ROOT / r["image_path"]).is_file():
            raise RuntimeError(f"missing file: {r['image_path']}")
        if verbose and len(rows) > _REPORT_EVERY and i % _REPORT_EVERY == 0:
            _log(f"[4/5] {i}/{len(rows)} paths ok", verbose=verbose)
    trains = {r["generator"] for r in rows if r["split"] == "train"}
    held = {r["generator"] for r in rows if r["split"] == "heldout"}
    assert held and not (held & trains), f"heldout leak into train: {held & trains}"
    write_manifest(rows, out)
    _log(f"[5/5] leak check vs {DEMO_DIR} ...", verbose=verbose)
    leak = check_demo_leak(rows, DEMO_DIR, REPO_ROOT, threshold, verbose=verbose)
    counts = Counter((r["split"], r["generator"]) for r in rows)
    print("split/generator rows:")
    for (s, g) in sorted(counts, key=lambda t: (SPLIT_ORDER.index(t[0]), t[1])):
        print(f"  {s:<8} {g:<12} {counts[(s, g)]}")
    print(f"total={len(rows)} dedupe_removed={removed} leak={leak} out={out} "
          f"({time.perf_counter() - t_all:.1f}s)")
    return {"rows": len(rows), "removed": removed, "leak": leak,
            "out": str(out), "heldout": sorted(held)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build data/processed/manifest.csv (Day 1 Block 1)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="scale total rows down (0=full ~27k)")
    ap.add_argument("--sid-max", type=int, default=60000)
    ap.add_argument("--threshold", type=int, default=5, help="pHash hamming threshold")
    ap.add_argument("--quiet", action="store_true", help="suppress progress lines")
    a = ap.parse_args(argv)
    build_manifest(seed=a.seed, out=a.out, limit=a.limit, sid_max=a.sid_max,
                   threshold=a.threshold, verbose=not a.quiet)


if __name__ == "__main__":
    main()

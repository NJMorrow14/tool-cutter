#!/usr/bin/env python3
"""
Sweep parameter combinations for tool_image_to_svg.py and generate multiple SVG outputs
to visually compare results. Produces an index.html and a CSV log of runs.

Usage example:
    python tool_image_to_svg.py -i overhead_dark.jpeg \
        -o cutouts_canny.svg \
        --save-debug cutouts_canny_mask.png \
        --mode canny --no-invert --blur-ksize 0 \
        --epsilon-frac 0.008074226794491369 
        --min-area 1200 
        --morph-close 0 \
        --morph-open 0 
        --dilate-iter 0 
        --erode-iter 1 
        --canny 100 245


 python sweep_svg_params.py \
    --input overhead_lime.jpeg \
    --outdir sweep_canny_local \
    --iters 200 \
    --seed 1 \
    --include-masks \
    --modes canny \
    --invert-fixed false \
    --blur-choices 0 3 \
    --epsilon-frac-range 0.006 0.010 \
    --min-area-range 1000 1600 \
    --morph-close-choices 0 3 \
    --morph-open-choices 0 3 \
    --dilate-range 0 1 \
    --erode-range 0 2 \
    --canny-low-range 90 110 \
    --canny-high-range 235 255

This script calls tool_image_to_svg.py in a loop with randomized but reasonable
parameters for each selected mode.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


TOOL_SCRIPT = Path(__file__).with_name("tool_image_to_svg.py")


def odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


@dataclass
class RunConfig:
    mode: str
    blur_ksize: int
    invert: bool
    epsilon_frac: float
    min_area: int
    morph_close: int
    morph_open: int
    dilate_iter: int
    erode_iter: int
    # Threshold mode
    threshold: int | None = None  # -1 for Otsu or 0..255
    # Adaptive mode
    adaptive_block_size: int | None = None
    adaptive_C: int | None = None
    # Canny mode
    canny_low: int | None = None
    canny_high: int | None = None

    def to_args(self) -> List[str]:
        args: List[str] = ["--mode", self.mode]
        if self.blur_ksize:
            args += ["--blur-ksize", str(self.blur_ksize)]
        args += (["--invert"] if self.invert else ["--no-invert"])
        args += ["--epsilon-frac", f"{self.epsilon_frac:.4f}"]
        args += ["--min-area", str(self.min_area)]
        if self.morph_close:
            args += ["--morph-close", str(self.morph_close)]
        if self.morph_open:
            args += ["--morph-open", str(self.morph_open)]
        if self.dilate_iter:
            args += ["--dilate-iter", str(self.dilate_iter)]
        if self.erode_iter:
            args += ["--erode-iter", str(self.erode_iter)]
        if self.mode == "threshold":
            if self.threshold is not None:
                args += ["--threshold", str(self.threshold)]
        elif self.mode == "adaptive":
            if self.adaptive_block_size is not None:
                args += ["--adaptive-block-size", str(self.adaptive_block_size)]
            if self.adaptive_C is not None:
                args += ["--adaptive-C", str(self.adaptive_C)]
        elif self.mode == "canny":
            if self.canny_low is not None and self.canny_high is not None:
                args += ["--canny", str(self.canny_low), str(self.canny_high)]
        return args

@dataclass
class SweepRanges:
    invert_fixed: Optional[bool] = None
    blur_choices: Optional[List[int]] = None
    epsilon_frac_range: Optional[Tuple[float, float]] = None
    min_area_range: Optional[Tuple[int, int]] = None
    morph_close_choices: Optional[List[int]] = None
    morph_open_choices: Optional[List[int]] = None
    dilate_range: Optional[Tuple[int, int]] = None
    erode_range: Optional[Tuple[int, int]] = None
    canny_low_range: Optional[Tuple[int, int]] = None
    canny_high_range: Optional[Tuple[int, int]] = None


def random_config(rng: random.Random, mode: str, ranges: Optional[SweepRanges] = None) -> RunConfig:
    ranges = ranges or SweepRanges()
    blur_options = ranges.blur_choices or [0, 3, 5, 7, 9]
    invert = ranges.invert_fixed if ranges.invert_fixed is not None else rng.choice([True, False])
    if ranges.epsilon_frac_range:
        e_min, e_max = ranges.epsilon_frac_range
        epsilon_frac = rng.uniform(float(e_min), float(e_max))
    else:
        epsilon_frac = rng.uniform(0.004, 0.02)
    if ranges.min_area_range:
        a_min, a_max = ranges.min_area_range
        min_area = rng.randint(int(a_min), int(a_max))
    else:
        min_area = rng.choice([500, 800, 1200, 2000, 3000, 5000, 8000])
    morph_close = rng.choice(ranges.morph_close_choices or [0, 3, 5, 7, 9, 11])
    morph_open = rng.choice(ranges.morph_open_choices or [0, 3, 5])
    if ranges.dilate_range:
        d_min, d_max = ranges.dilate_range
        dilate_iter = rng.randint(int(d_min), int(d_max))
    else:
        dilate_iter = rng.choice([0, 1, 2])
    if ranges.erode_range:
        e_min, e_max = ranges.erode_range
        erode_iter = rng.randint(int(e_min), int(e_max))
    else:
        erode_iter = rng.choice([0, 1, 2])

    if mode == "threshold":
        thr = rng.choice([-1, 60, 90, 110, 140, 170, 200])
        return RunConfig(
            mode=mode,
            blur_ksize=rng.choice(blur_options),
            invert=invert,
            epsilon_frac=epsilon_frac,
            min_area=min_area,
            morph_close=morph_close,
            morph_open=morph_open,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
            threshold=thr,
        )
    elif mode == "adaptive":
        bs = odd(rng.randrange(21, 61, 2))
        C = rng.randrange(0, 11)
        return RunConfig(
            mode=mode,
            blur_ksize=rng.choice(blur_options),
            invert=invert,
            epsilon_frac=epsilon_frac,
            min_area=min_area,
            morph_close=morph_close,
            morph_open=morph_open,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
            adaptive_block_size=bs,
            adaptive_C=C,
        )
    else:  # canny
        if ranges.canny_low_range:
            cl_min, cl_max = ranges.canny_low_range
            low = rng.randint(int(cl_min), int(cl_max))
        else:
            low = rng.randrange(30, 101, 5)
        if ranges.canny_high_range:
            ch_min, ch_max = ranges.canny_high_range
            lo = max(int(low) + 1, int(ch_min))
            if lo > int(ch_max):
                lo = int(ch_max)
            high = rng.randint(lo, int(ch_max))
        else:
            high = low + rng.randrange(60, 151, 5)
        return RunConfig(
            mode=mode,
            blur_ksize=rng.choice(blur_options),
            invert=invert,
            epsilon_frac=epsilon_frac,
            min_area=min_area,
            morph_close=rng.choice([0, 3, 5, 7, 9, 11]),
            morph_open=morph_open,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
            canny_low=low,
            canny_high=high,
        )


def ensure_tool_exists():
    if not TOOL_SCRIPT.exists():
        print(f"ERROR: {TOOL_SCRIPT} not found. Make sure tool_image_to_svg.py is in the same directory.")
        sys.exit(1)


def build_cmd(img: Path, out_svg: Path, cfg: RunConfig, save_mask: Path | None) -> List[str]:
    cmd = [sys.executable, str(TOOL_SCRIPT), "-i", str(img), "-o", str(out_svg)]
    cmd += cfg.to_args()
    if save_mask is not None:
        cmd += ["--save-debug", str(save_mask)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", required=True, help="Input image path")
    ap.add_argument("--outdir", "-o", required=True, help="Output directory for sweep results")
    ap.add_argument("--iters", type=int, default=50, help="Number of runs to generate")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["threshold", "adaptive", "canny"],
        choices=["threshold", "adaptive", "canny"],
        help="Which modes to include in the sweep",
    )
    ap.add_argument("--include-masks", action="store_true", help="Also save the binary mask images for each run")
    # One-factor-at-a-time mode and baseline
    ap.add_argument("--ofat", action="store_true", help="Vary one parameter at a time around a baseline")
    ap.add_argument("--steps-per-param", type=int, default=None, help="If --ofat: steps for each parameter (defaults to iters/num_params)")
    ap.add_argument("--baseline-invert", type=str, choices=["true", "false"], default=None)
    ap.add_argument("--baseline-blur", type=int, default=None)
    ap.add_argument("--baseline-epsilon-frac", type=float, default=None)
    ap.add_argument("--baseline-min-area", type=int, default=None)
    ap.add_argument("--baseline-morph-close", type=int, default=None)
    ap.add_argument("--baseline-morph-open", type=int, default=None)
    ap.add_argument("--baseline-dilate-iter", type=int, default=None)
    ap.add_argument("--baseline-erode-iter", type=int, default=None)
    ap.add_argument("--baseline-threshold", type=int, default=None)
    ap.add_argument("--baseline-adaptive-block-size", type=int, default=None)
    ap.add_argument("--baseline-adaptive-C", type=int, default=None)
    ap.add_argument("--baseline-canny-low", type=int, default=None)
    ap.add_argument("--baseline-canny-high", type=int, default=None)
    # Local sweep controls near a baseline (also used by random sweep)
    ap.add_argument("--invert-fixed", type=str, choices=["true", "false"], default=None, help="Fix invert to true/false instead of random")
    ap.add_argument("--blur-choices", type=int, nargs="+", default=None, help="Choices for blur kernel size, e.g. 0 3 5")
    ap.add_argument("--epsilon-frac-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None, help="Range for epsilon-frac")
    ap.add_argument("--min-area-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None, help="Range for min-area")
    ap.add_argument("--morph-close-choices", type=int, nargs="+", default=None, help="Choices for morph-close, e.g. 0 3 5")
    ap.add_argument("--morph-open-choices", type=int, nargs="+", default=None, help="Choices for morph-open")
    ap.add_argument("--dilate-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None, help="Range for dilate-iter")
    ap.add_argument("--erode-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None, help="Range for erode-iter")
    ap.add_argument("--canny-low-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None, help="Range for Canny low")
    ap.add_argument("--canny-high-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None, help="Range for Canny high")
    args = ap.parse_args()

    ensure_tool_exists()

    img = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    base = img.stem

    # Prepare parameter ranges/choices used for sampling
    ranges = SweepRanges(
        invert_fixed=(None if args.invert_fixed is None else (args.invert_fixed == "true")),
        blur_choices=args.blur_choices,
        epsilon_frac_range=tuple(args.epsilon_frac_range) if args.epsilon_frac_range else None,
        min_area_range=tuple(args.min_area_range) if args.min_area_range else None,
        morph_close_choices=args.morph_close_choices,
        morph_open_choices=args.morph_open_choices,
        dilate_range=tuple(args.dilate_range) if args.dilate_range else None,
        erode_range=tuple(args.erode_range) if args.erode_range else None,
        canny_low_range=tuple(args.canny_low_range) if args.canny_low_range else None,
        canny_high_range=tuple(args.canny_high_range) if args.canny_high_range else None,
    )

    # Prepare logs
    log_path = outdir / "sweep_log.csv"
    html_path = outdir / "index.html"

    fieldnames: List[str] = [
        "index",
        "status",
        "svg_path",
        "mask_path",
        "mode",
        "blur_ksize",
        "invert",
        "epsilon_frac",
        "min_area",
        "morph_close",
        "morph_open",
        "dilate_iter",
        "erode_iter",
        "threshold",
        "adaptive_block_size",
        "adaptive_C",
        "canny_low",
        "canny_high",
    ]

    rows: List[Dict[str, str]] = []

    # Build runs list (OFAT or randomized)
    runs: List[RunConfig] = []

    if args.ofat:
        # Enforce a single mode
        if len(args.modes) != 1:
            print("Error: --ofat requires exactly one mode via --modes", file=sys.stderr)
            sys.exit(2)
        mode = args.modes[0]
        # Baseline
        base_cfg = RunConfig(
            mode=mode,
            blur_ksize=int(args.baseline_blur if args.baseline_blur is not None else (ranges.blur_choices[0] if ranges.blur_choices else 0)),
            invert=(False if args.baseline_invert == "false" else True if args.baseline_invert == "true" else (ranges.invert_fixed if ranges.invert_fixed is not None else True)),
            epsilon_frac=float(args.baseline_epsilon_frac if args.baseline_epsilon_frac is not None else (sum(ranges.epsilon_frac_range)/2 if ranges.epsilon_frac_range else 0.01)),
            min_area=int(args.baseline_min_area if args.baseline_min_area is not None else (sum(ranges.min_area_range)//2 if ranges.min_area_range else 1200)),
            morph_close=int(args.baseline_morph_close if args.baseline_morph_close is not None else (ranges.morph_close_choices[0] if ranges.morph_close_choices else 0)),
            morph_open=int(args.baseline_morph_open if args.baseline_morph_open is not None else (ranges.morph_open_choices[0] if ranges.morph_open_choices else 0)),
            dilate_iter=int(args.baseline_dilate_iter if args.baseline_dilate_iter is not None else (ranges.dilate_range[0] if ranges.dilate_range else 0)),
            erode_iter=int(args.baseline_erode_iter if args.baseline_erode_iter is not None else (ranges.erode_range[0] if ranges.erode_range else 1)),
        )
        if mode == "threshold":
            base_cfg.threshold = int(args.baseline_threshold if args.baseline_threshold is not None else -1)
        elif mode == "adaptive":
            base_cfg.adaptive_block_size = int(args.baseline_adaptive_block_size if args.baseline_adaptive_block_size is not None else 35)
            base_cfg.adaptive_C = int(args.baseline_adaptive_C if args.baseline_adaptive_C is not None else 5)
        else:
            base_cfg.canny_low = int(args.baseline_canny_low if args.baseline_canny_low is not None else (ranges.canny_low_range[0] if ranges.canny_low_range else 100))
            base_cfg.canny_high = int(args.baseline_canny_high if args.baseline_canny_high is not None else (ranges.canny_high_range[1] if ranges.canny_high_range else 245))

        # Collect parameters to sweep
        staged_values: List[Tuple[str, List]] = []
        staged_ranges: List[Tuple[str, Tuple, bool]] = []  # (name, (lo, hi), is_float)
        if ranges.blur_choices is not None:
            staged_values.append(("blur_ksize", ranges.blur_choices))
        if args.baseline_invert is None and ranges.invert_fixed is None:
            staged_values.append(("invert", [True, False]))
        if ranges.epsilon_frac_range:
            staged_ranges.append(("epsilon_frac", ranges.epsilon_frac_range, True))
        if ranges.min_area_range:
            staged_ranges.append(("min_area", ranges.min_area_range, False))
        if ranges.morph_close_choices is not None:
            staged_values.append(("morph_close", ranges.morph_close_choices))
        if ranges.morph_open_choices is not None:
            staged_values.append(("morph_open", ranges.morph_open_choices))
        if ranges.dilate_range:
            staged_ranges.append(("dilate_iter", ranges.dilate_range, False))
        if ranges.erode_range:
            staged_ranges.append(("erode_iter", ranges.erode_range, False))
        if mode == "canny":
            if ranges.canny_low_range:
                staged_ranges.append(("canny_low", ranges.canny_low_range, False))
            if ranges.canny_high_range:
                staged_ranges.append(("canny_high", ranges.canny_high_range, False))

        num_params = len(staged_values) + len(staged_ranges)
        if num_params == 0:
            print("Warning: --ofat set but no ranges/choices provided; falling back to randomized.")
        else:
            steps = args.steps_per_param or max(1, args.iters // num_params)
            def linspace_int(a: int, b: int, n: int) -> List[int]:
                if n <= 1:
                    return [a]
                return [int(round(a + (b - a) * t / (n - 1))) for t in range(n)]
            def linspace_float(a: float, b: float, n: int) -> List[float]:
                if n <= 1:
                    return [a]
                return [a + (b - a) * t / (n - 1) for t in range(n)]
            for name, (lo, hi), is_float in staged_ranges:
                vals = linspace_float(float(lo), float(hi), steps) if is_float else linspace_int(int(lo), int(hi), steps)
                staged_values.append((name, vals))
            # Build runs: vary one param at a time
            for name, vals in staged_values:
                for v in vals:
                    cfg = RunConfig(**asdict(base_cfg))
                    setattr(cfg, name, v)
                    if mode == "canny" and name in ("canny_low", "canny_high"):
                        if cfg.canny_high is not None and cfg.canny_low is not None and cfg.canny_high <= cfg.canny_low:
                            continue
                    runs.append(cfg)
            # Conform to requested total
            if len(runs) > args.iters:
                runs = runs[: args.iters]
            elif len(runs) < args.iters and len(runs) > 0:
                k = args.iters - len(runs)
                runs.extend(runs[:k])

    if not runs:
        # Randomized sweep as fallback or when --ofat not set
        for _ in range(args.iters):
            mode = rng.choice(args.modes)
            cfg = random_config(rng, mode, ranges)
            runs.append(cfg)

    # Execute runs
    for i, cfg in enumerate(runs):
        name_bits = [
            f"{base}",
            f"i{i:03d}",
            f"{cfg.mode}",
            f"b{cfg.blur_ksize}",
            ("inv" if cfg.invert else "noinv"),
            f"e{cfg.epsilon_frac:.3f}",
            f"a{cfg.min_area}",
            f"mc{cfg.morph_close}",
            f"mo{cfg.morph_open}",
            f"d{cfg.dilate_iter}",
            f"er{cfg.erode_iter}",
        ]
        if cfg.mode == "threshold":
            name_bits.append(f"thr{cfg.threshold}")
        elif cfg.mode == "adaptive":
            name_bits.append(f"bs{cfg.adaptive_block_size}_C{cfg.adaptive_C}")
        else:
            name_bits.append(f"c{cfg.canny_low}-{cfg.canny_high}")

        base_name = "__".join(name_bits)
        if len(base_name) > 180:
            base_name = base_name[:180]

        out_svg = outdir / f"{base_name}.svg"
        out_mask = (outdir / f"{base_name}_mask.png") if args.include_masks else None

        cmd = build_cmd(img, out_svg, cfg, out_mask)

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            status = "ok" if res.returncode == 0 and out_svg.exists() else f"error({res.returncode})"
            if status != "ok":
                err_first = (res.stderr or res.stdout).splitlines()[:2]
                status += ":" + " | ".join(err_first)
        except Exception as e:  # noqa: BLE001
            status = f"exception:{e}"

        row: Dict[str, str] = {
            "index": str(i),
            "status": status,
            "svg_path": os.path.relpath(out_svg, outdir) if out_svg.exists() else "",
            "mask_path": os.path.relpath(out_mask, outdir) if out_mask and out_mask.exists() else "",
            **{k: str(v) for k, v in asdict(cfg).items()},
        }
        rows.append(row)

    # Write CSV
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Build simple HTML gallery
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<title>SVG Sweep Results</title>")
    parts.append("<style>body{font-family:sans-serif;margin:16px;} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;} .card{border:1px solid #ddd;padding:8px;border-radius:8px;} .row{display:flex;gap:8px;align-items:flex-start;} .thumb{width:100%; height:220px; border:1px solid #ccc; background:#fafafa; display:flex; align-items:center; justify-content:center;} .thumb img, .thumb object{max-width:100%; max-height:100%;} .meta{font-size:12px; color:#333; white-space:pre-wrap;}</style>")
    parts.append(f"<h1>SVG Sweep Results</h1><p>Created: {created}</p>")
    parts.append("<div class='grid'>")
    for r in rows:
        if r["status"].startswith("ok") and r["svg_path"]:
            meta = []
            meta.append(f"mode={r['mode']}, invert={r['invert']}")
            meta.append(f"blur={r['blur_ksize']}, eps={r['epsilon_frac']}, min_area={r['min_area']}")
            meta.append(f"mc={r['morph_close']}, mo={r['morph_open']}, d={r['dilate_iter']}, e={r['erode_iter']}")
            if r['mode'] == 'threshold':
                meta.append(f"threshold={r['threshold']}")
            elif r['mode'] == 'adaptive':
                meta.append(f"block={r['adaptive_block_size']}, C={r['adaptive_C']}")
            else:
                meta.append(f"canny={r['canny_low']}-{r['canny_high']}")
            mask_html = f"<div class='thumb'><img src='{r['mask_path']}'/></div>" if r["mask_path"] else ""
            parts.append(
                "<div class='card'>"
                f"<div class='thumb'><object type='image/svg+xml' data='{r['svg_path']}'></object></div>"
                f"<div class='row'>{mask_html}</div>"
                f"<div class='meta'>{' | '.join(meta)}</div>"
                "</div>"
            )
        else:
            # Show errors as cards too for visibility
            parts.append(
                "<div class='card'>"
                f"<div class='meta'>Run {r['index']} failed: {r['status']}</div>"
                "</div>"
            )
    parts.append("</div>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print(f"Wrote {len(rows)} runs\n- CSV: {log_path}\n- HTML: {html_path}\nOpen index.html in a browser to compare results.")


if __name__ == "__main__":
    main()

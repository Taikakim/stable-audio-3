#!/usr/bin/env python3
"""Master Shootout Comparison Compiler for SA3 Drum Separation Scratch Ranks.

Aggregates eval_results.json across all experimental arms:
- Arm 1: Rank 64 NorMuon + SF (SF-NorMuon on 33M inlet, SF-AdamW on LoRA)
- Arm 0: Rank 64 AdamW Control (WSD schedule, 2e-4)
- Arm 2: Rank 64 Baseline Muon (Pure NS5, 3e-4, WSD schedule)
- Arm 3: Rank 128 NorMuon + SF (SF-NorMuon on inlet, SF-AdamW on rank-128 LoRA)
- Reference: Step 6000 Baseline (Rank 32 AdamW)

Generates:
- summary_table.json
- index.html (Master Shootout Console)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

def build_shootout_dashboard(eval_root: Path):
    arms_meta = {
        "arm1_r64_normuon_sf": {
            "name": "Arm 1: Rank 64 NorMuon + SF (Step 2000)",
            "tag": "NorMuon + SF",
            "rank": 64,
            "params": "117.6M",
            "desc": "33M mix_lat: SF-NorMuon (1e-3) | LoRA: SF-AdamW (2e-4) | Step 2000 (Loss 0.167)",
            "color": "#38bdf8"
        },
        "arm0_r64_adamw": {
            "name": "Arm 0: Rank 64 AdamW Control",
            "tag": "AdamW Control",
            "rank": 64,
            "params": "117.6M",
            "desc": "All params: AdamW (2e-4) | WSD schedule (knee 25%, decay to 0.1x)",
            "color": "#94a3b8"
        },
        "arm2_r64_muon_baseline": {
            "name": "Arm 2: Rank 64 Baseline Muon",
            "tag": "Baseline Muon",
            "rank": 64,
            "params": "117.6M",
            "desc": "33M mix_lat: Pure NS5 (3e-4) | LoRA: AdamW (2e-4) | WSD decay",
            "color": "#f59e0b"
        },
        "arm3_r128_normuon_sf": {
            "name": "Arm 3: Rank 128 NorMuon + SF",
            "tag": "Rank 128 NorMuon + SF",
            "rank": 128,
            "params": "202.2M",
            "desc": "33M mix_lat: SF-NorMuon (1e-3) | Rank-128 LoRA: force-scalar SF-AdamW (2e-4)",
            "color": "#a855f7"
        }
    }

    results = {}
    for arm_id, meta in arms_meta.items():
        res_file = eval_root / arm_id / "eval_results.json"
        if res_file.exists():
            with open(res_file) as f:
                results[arm_id] = {
                    "meta": meta,
                    "data": json.load(f)
                }

    # Reference baseline (step 6000) if available
    ref_file = Path("/home/kim/sao_runs/drum_sep_lr_brackets_eval/baseline_step6000/eval_results.json")
    if ref_file.exists():
        with open(ref_file) as f:
            results["baseline_step6000"] = {
                "meta": {
                    "name": "Prior Baseline (Step 6000)",
                    "tag": "Rank 32 Step-6000",
                    "rank": 32,
                    "params": "74.5M",
                    "desc": "Prior run: Rank 32 AdamW 1e-4 trained for 6000 steps",
                    "color": "#64748b"
                },
                "data": json.load(f)
            }

    # Extract stems breakdown
    stem_types = ["kick", "snare", "hihat", "percussion"]

    rows_html = []
    for arm_id, item in results.items():
        m = item["meta"]
        d = item["data"]
        val_samples = d.get("val_samples", [])
        comp_samples = d.get("comp_samples", [])

        mean_sisdr = d.get("mean_val_si_sdr_db", float("nan"))
        mean_wsisdr = d.get("mean_val_wiener_si_sdr_db", float("nan"))
        mean_lsd = d.get("mean_val_lsd_db", float("nan"))
        mean_f1 = d.get("mean_val_onset_f1_20", float("nan"))

        # Per stem SI-SDR
        stem_sisdr = {}
        for st in stem_types:
            st_samples = [s for s in val_samples if f"_{st}_" in s.get("sample_tag", "") or st == s.get("stem_tag", "") or st == s.get("stem", "")]
            if st_samples:
                stem_sisdr[st] = float(np.mean([s.get("wiener", {}).get("si_sdr_db", s.get("si_sdr_db", float("nan"))) for s in st_samples]))
            else:
                stem_sisdr[st] = float("nan")

        comp_sisdr = float(np.mean([s["sa3"]["si_sdr_db"] for s in comp_samples])) if comp_samples else float("nan")
        comp_wsisdr = float(np.mean([s["sa3_wiener"]["si_sdr_db"] for s in comp_samples if "sa3_wiener" in s])) if comp_samples else float("nan")

        console_url = f"{arm_id}/index.html"

        rows_html.append(f"""
        <tr style="border-bottom: 1px solid #1e293b;">
          <td style="padding: 14px 16px;">
            <div style="font-weight: 700; color: {m['color']}; font-size: 15px;">{m['name']}</div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">{m['desc']}</div>
          </td>
          <td style="padding: 14px 16px; text-align: center; font-family: monospace;">{m['rank']}</td>
          <td style="padding: 14px 16px; text-align: center; font-family: monospace;">{m['params']}</td>
          <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-size: 16px; color: {'#10b981' if mean_wsisdr > -5 else '#f8fafc'};">
            {mean_wsisdr:+.2f} dB
            <div style="font-size: 11px; color: #94a3b8; font-weight: normal;">raw: {mean_sisdr:+.2f} dB</div>
          </td>
          <td style="padding: 14px 16px; text-align: right; font-family: monospace;">{mean_lsd:.2f} dB</td>
          <td style="padding: 14px 16px; text-align: right; font-family: monospace;">{mean_f1:.3f}</td>
          <td style="padding: 14px 16px; text-align: right; font-family: monospace;">{stem_sisdr.get('kick', float('nan')):+.1f}</td>
          <td style="padding: 14px 16px; text-align: right; font-family: monospace;">{stem_sisdr.get('snare', float('nan')):+.1f}</td>
          <td style="padding: 14px 16px; text-align: right; font-family: monospace; color: {'#ef4444' if stem_sisdr.get('hihat', -99) < -25 else '#10b981'};">
            {stem_sisdr.get('hihat', float('nan')):+.1f}
          </td>
          <td style="padding: 14px 16px; text-align: right; font-family: monospace;">{stem_sisdr.get('percussion', float('nan')):+.1f}</td>
          <td style="padding: 14px 16px; text-align: center;">
            <a href="{console_url}" style="display: inline-block; padding: 6px 12px; background: #1e293b; color: #38bdf8; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 12px; border: 1px solid #334155;">Audition &rarr;</a>
          </td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SA3 Drum Separation — From Scratch Ranks & Optimizer Shootout</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #070b14;
      --card-bg: #0f172a;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 32px 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1350px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      text-align: left;
      padding: 12px 16px;
      background: #131d31;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
      border-bottom: 2px solid var(--border);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>SA3 Drum Separation Shootout: Ranks 64/128 & Modular Optimizer</h1>
      <p style="color: var(--text-muted); font-size: 14px; margin-top: 6px;">
        Trained completely from scratch from frozen <code>medium-base</code> on 19,278 paired drum crops.
        Comparing <strong>NorMuon + Schedule-Free</strong> vs <strong>Baseline Muon</strong> vs <strong>AdamW</strong>, scaling across <strong>Rank 64</strong> and <strong>Rank 128</strong>.
      </p>
    </header>

    <div class="card" style="padding: 0; overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Experimental Arm</th>
            <th style="text-align: center;">Rank</th>
            <th style="text-align: center;">Params</th>
            <th style="text-align: right;">Val Wiener SI-SDR</th>
            <th style="text-align: right;">Val LSD</th>
            <th style="text-align: right;">Onset F1</th>
            <th style="text-align: right;">Kick</th>
            <th style="text-align: right;">Snare</th>
            <th style="text-align: right;">Hi-Hat</th>
            <th style="text-align: right;">Perc</th>
            <th style="text-align: center;">Console</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html) if rows_html else '<tr><td colspan="11" style="text-align:center; padding: 24px; color: #94a3b8;">No completed arms yet. Training in progress...</td></tr>'}
        </tbody>
      </table>
    </div>

    <div style="font-size: 12px; color: #64748b; text-align: center; margin-top: 24px;">
      Generated by ANTIGRAVITY • Serving on port 8088 • SA3 Gate G3 Multi-Stem Drum Separator
    </div>
  </div>
</body>
</html>
"""

    out_file = eval_root / "index.html"
    out_file.write_text(html)
    print(f"[compile_ranks_shootout] Master dashboard updated: file://{out_file}")

if __name__ == "__main__":
    eval_dir = Path("/home/kim/sao_runs/drum_sep_scratch_ranks_eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    build_shootout_dashboard(eval_dir)

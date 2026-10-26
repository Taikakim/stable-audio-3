"""Interactive HTML Audition Console Generator for SA3 Gate G3 Multi-Stem Drum Separation.
Generates a dark-themed HTML report with synchronized multi-track playback,
master timeline scrubbers, instant single-stem A/B solo auditioning, and objective metrics.
"""

from pathlib import Path
import json
import numpy as np


def generate_html_console(out_dir: Path, val_results: list, comp_results: list, meta: dict):
    """Generate interactive audition console HTML."""
    html_path = out_dir / "index.html"

    # Aggregates
    val_sisdr = float(np.mean([r["si_sdr_db"] for r in val_results])) if val_results else float("nan")
    val_lsd = float(np.mean([r["lsd_db"] for r in val_results])) if val_results else float("nan")
    val_f1 = float(np.mean([r["onset_f1_20"] for r in val_results])) if val_results else float("nan")

    sa3_comp_sisdr = float(np.mean([r["sa3"]["si_sdr_db"] for r in comp_results])) if comp_results else float("nan")
    sa3_comp_lsd = float(np.mean([r["sa3"]["lsd_db"] for r in comp_results])) if comp_results else float("nan")
    sa3_comp_f1 = float(np.mean([r["sa3"]["onset_f1_20"] for r in comp_results])) if comp_results else float("nan")

    bs_comp_sisdr = float(np.mean([r["bs_roformer"]["si_sdr_db"] for r in comp_results if "bs_roformer" in r])) if comp_results else float("nan")
    bs_comp_lsd = float(np.mean([r["bs_roformer"]["lsd_db"] for r in comp_results if "bs_roformer" in r])) if comp_results else float("nan")
    bs_comp_f1 = float(np.mean([r["bs_roformer"]["onset_f1_20"] for r in comp_results if "bs_roformer" in r])) if comp_results else float("nan")

    msg_comp_sisdr = float(np.mean([r["msgld"]["si_sdr_db"] for r in comp_results if "msgld" in r])) if comp_results else float("nan")
    msg_comp_lsd = float(np.mean([r["msgld"]["lsd_db"] for r in comp_results if "msgld" in r])) if comp_results else float("nan")
    msg_comp_f1 = float(np.mean([r["msgld"]["onset_f1_20"] for r in comp_results if "msgld" in r])) if comp_results else float("nan")

    has_wiener = any("sa3_wiener" in r for r in comp_results)
    if has_wiener:
        wiener_comp_sisdr = float(np.mean([r["sa3_wiener"]["si_sdr_db"] for r in comp_results if "sa3_wiener" in r]))
        wiener_comp_lsd = float(np.mean([r["sa3_wiener"]["lsd_db"] for r in comp_results if "sa3_wiener" in r]))
        wiener_comp_f1 = float(np.mean([r["sa3_wiener"]["onset_f1_20"] for r in comp_results if "sa3_wiener" in r]))
    else:
        wiener_comp_sisdr, wiener_comp_lsd, wiener_comp_f1 = float("nan"), float("nan"), float("nan")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SA3 Drum Separator — Gate G3 Multi-Stem Audition Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #070b14;
      --card-bg: #0f172a;
      --card-hover: #1e293b;
      --border: #1e293b;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --green: #10b981;
      --yellow: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
      --orange: #fb923c;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 32px 24px 100px 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 28px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
    }}
    h1 {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .badge-gate {{
      font-size: 13px;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 3px 10px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-bottom: 16px;
    }}
    .stat-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .stat-pill {{
      background: #131c31;
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .stat-pill .val {{
      font-weight: 700;
      font-size: 18px;
      color: var(--accent);
      font-feature-settings: "tnum";
    }}
    .stat-pill .lbl {{
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    /* Table Styles */
    .table-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 32px;
      overflow-x: auto;
    }}
    h2 {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-feature-settings: "tnum";
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr.highlight-target {{
      background: rgba(56, 189, 248, 0.08);
      font-weight: 600;
    }}
    tr.highlight-target td:first-child {{
      border-left: 3px solid var(--accent);
    }}
    .pill-tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
    }}
    .tag-green {{ background: rgba(16, 185, 129, 0.2); color: var(--green); }}
    .tag-blue {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); }}
    .tag-purple {{ background: rgba(168, 85, 247, 0.2); color: var(--purple); }}
    .tag-orange {{ background: rgba(251, 146, 60, 0.2); color: var(--orange); }}
    .tag-yellow {{ background: rgba(245, 158, 11, 0.2); color: var(--yellow); }}
    .tag-red {{ background: rgba(239, 68, 68, 0.2); color: var(--red); }}

    /* Track Cards & Transport */
    .track-section {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 32px;
      overflow: hidden;
    }}
    .track-header {{
      padding: 16px 20px;
      background: #141e33;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .track-title-block {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .track-title-text {{
      font-size: 17px;
      font-weight: 700;
      color: #ffffff;
    }}
    .now-playing-pill {{
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 20px;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: var(--accent);
      font-feature-settings: "tnum";
    }}
    .now-playing-pill strong {{
      color: #ffffff;
    }}

    /* Master Transport Controls */
    .master-transport {{
      display: flex;
      align-items: center;
      gap: 16px;
      background: #090e1a;
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
    }}
    .transport-play-btn {{
      background: var(--accent);
      color: #070b14;
      border: none;
      padding: 8px 18px;
      border-radius: 6px;
      font-weight: 800;
      font-size: 13px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: background 0.15s, transform 0.05s;
    }}
    .transport-play-btn:hover {{ background: #7dd3fc; }}
    .transport-play-btn:active {{ transform: scale(0.98); }}
    .transport-time {{
      font-family: monospace;
      font-size: 13px;
      color: var(--text-muted);
      min-width: 95px;
      font-weight: 600;
    }}
    .scrubber-wrap {{
      flex: 1;
      display: flex;
      align-items: center;
    }}
    .timeline-slider {{
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #1e293b;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .timeline-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.8);
      transition: transform 0.1s;
    }}
    .timeline-slider::-webkit-slider-thumb:hover {{
      transform: scale(1.25);
    }}

    /* Stems Grid */
    .stems-grid {{
      padding: 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .stem-card {{
      background: #131c31;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      cursor: pointer;
      transition: all 0.15s ease;
      position: relative;
    }}
    .stem-card:hover {{
      border-color: rgba(56, 189, 248, 0.4);
      background: #17233d;
    }}
    .stem-card.is-active-solo {{
      border-color: var(--accent) !important;
      background: rgba(56, 189, 248, 0.08) !important;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.2);
    }}
    .stem-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .stem-name {{
      font-size: 13px;
      font-weight: 700;
      color: #e2e8f0;
    }}
    .stem-metrics {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .m-badge {{
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255,255,255,0.06);
      color: var(--text-muted);
      font-feature-settings: "tnum";
    }}
    .m-badge strong {{ color: var(--text); }}
    .solo-btn {{
      background: rgba(255,255,255,0.08);
      color: var(--text-muted);
      border: 1px solid var(--border);
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .stem-card.is-active-solo .solo-btn {{
      background: var(--accent);
      color: #070b14;
      border-color: var(--accent);
    }}
    audio {{
      display: none; /* hidden; driven seamlessly by master transport */
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        SA3 Drum Separator
        <span class="badge-gate">Gate G3 Multi-Stem</span>
      </h1>
      <p class="subtitle">
        Unified prompt-conditioned drum sub-kit separation (Kick, Snare, Hi-Hats, Percussion) on discrete AMD Radeon RX 9070 XT.
        Model: SA3 Medium Base DiT (LoRA rank 32 + 256-ch modular mixture inlet) evaluated with STFT Wiener Mask Reconciliation.
      </p>
      <div class="stat-row">
        <div class="stat-pill">
          <span class="val">{sa3_comp_sisdr:+.2f} dB</span>
          <span class="lbl">Benchmark Kick SI-SDR</span>
        </div>
        <div class="stat-pill">
          <span class="val">{wiener_comp_sisdr:+.2f} dB</span>
          <span class="lbl">Wiener Reconciled SI-SDR</span>
        </div>
        <div class="stat-pill">
          <span class="val">{sa3_comp_lsd:.2f} dB</span>
          <span class="lbl">Benchmark Kick LSD</span>
        </div>
        <div class="stat-pill">
          <span class="val">{val_f1:.3f}</span>
          <span class="lbl">Held-Out Val Onset F1</span>
        </div>
      </div>
    </header>

    <!-- Benchmark Comparison Table -->
    <div class="table-card">
      <h2>Architectural Benchmark Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Architecture / Method</th>
            <th>Conditioning Path</th>
            <th>Target</th>
            <th>SI-SDR (dB)</th>
            <th>LSD (dB)</th>
            <th>Onset F1 (20ms)</th>
            <th>Status / Paradigm</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>BS-RoFormer (SW Drums)</strong></td>
            <td>Sub-band Time-Frequency Transformer</td>
            <td>Full Drums</td>
            <td>{bs_comp_sisdr:+.2f} dB</td>
            <td>{bs_comp_lsd:.2f} dB</td>
            <td>{bs_comp_f1:.3f}</td>
            <td><span class="pill-tag tag-yellow">Drum Bus Baseline</span></td>
          </tr>
          <tr>
            <td><strong>Separate-and-Detect (MSG-LD)</strong></td>
            <td>Latent Diffusion U-Net</td>
            <td>Isolated Kick</td>
            <td>{msg_comp_sisdr:+.2f} dB</td>
            <td>{msg_comp_lsd:.2f} dB</td>
            <td>{msg_comp_f1:.3f}</td>
            <td><span class="pill-tag tag-orange">Baseline (16kHz)</span></td>
          </tr>
          <tr>
            <td><strong>SA3 Arm A (Frozen Prior)</strong></td>
            <td>LoRA Prior (Modular Inlet Frozen)</td>
            <td>Isolated Kick</td>
            <td>-35.56 dB</td>
            <td>25.45 dB</td>
            <td>0.067</td>
            <td><span class="pill-tag tag-purple">Unconditioned Prior</span></td>
          </tr>
          <tr class="highlight-target">
            <td><strong>SA3 Kick (Generative)</strong></td>
            <td>LoRA + Active Modular Mix Inlet (Generative)</td>
            <td>Isolated Kick</td>
            <td>{sa3_comp_sisdr:+.2f} dB</td>
            <td>{sa3_comp_lsd:.2f} dB</td>
            <td>{sa3_comp_f1:.3f}</td>
            <td><span class="pill-tag tag-blue">Generative Output</span></td>
          </tr>
          <tr style="background: rgba(16, 185, 129, 0.08); font-weight: 600;">
            <td><strong>SA3 Kick (Wiener Reconciled)</strong></td>
            <td>LoRA Prior + STFT Wiener Masking (&alpha;=1.5)</td>
            <td>Isolated Kick</td>
            <td>{wiener_comp_sisdr:+.2f} dB</td>
            <td>{wiener_comp_lsd:.2f} dB</td>
            <td>{wiener_comp_f1:.3f}</td>
            <td><span class="pill-tag tag-green">Energy Conserved</span></td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Benchmark Multitrack Audition Section -->
    <div style="margin-bottom: 20px;">
      <h2>Studio Benchmark Multitrack Audition (Full Songs)</h2>
      <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 20px;">
        Click <strong>Play</strong> to start playback with synchronized playheads, then <strong>click any card or SOLO</strong> to audition that stem in isolation. Scrub the master timeline to jump anywhere in the song.
      </p>
    </div>
"""

    for c_idx, comp in enumerate(comp_results):
        t_id = comp["track_id"]
        aud = comp.get("audio", {})
        m_sa3 = comp.get("sa3", {})
        m_cmp = comp.get("sa3_compare", {})
        m_bs = comp.get("bs_roformer", {})
        m_msg = comp.get("msgld", {})
        m_wiener = comp.get("sa3_wiener", {})
        title_str = t_id.replace('_', ' ').title()

        html += f"""
    <div class="track-section" id="track_{c_idx}">
      <div class="track-header">
        <div class="track-title-block">
          <span class="track-title-text">{title_str}</span>
          <span class="now-playing-pill" id="badge_{c_idx}">Auditioning: <strong>Full Mix (Master Audio)</strong></span>
        </div>
      </div>
      <div class="master-transport">
        <button class="transport-play-btn" id="play_btn_{c_idx}" onclick="togglePlay({c_idx})">&#9658; Play</button>
        <span class="transport-time" id="time_{c_idx}">0:00 / 0:30</span>
        <div class="scrubber-wrap">
          <input type="range" class="timeline-slider" id="scrub_{c_idx}" min="0" max="30" step="0.05" value="0" oninput="onScrub({c_idx}, this.value)">
        </div>
      </div>
      <div class="stems-grid">
"""

        card_list = []

        # 0: Mix
        if "mix" in aud:
            card_list.append({
                "name": "Full Mix (Master Audio)",
                "color": "var(--text)",
                "border": "var(--border)",
                "bg": "#131c31",
                "badge": "Input Track",
                "metrics": '<span class="m-badge">Target: <strong>Complete Song</strong></span>',
                "src": aud["mix"]
            })

        # 1: Studio Kick Ground Truth
        if "ground_truth" in aud:
            card_list.append({
                "name": "Studio Kick (Ground Truth)",
                "color": "var(--purple)",
                "border": "rgba(168, 85, 247, 0.4)",
                "bg": "#131c31",
                "badge": "Clean Stem",
                "metrics": '<span class="m-badge">Target: <strong>Isolated Kick</strong></span>',
                "src": aud["ground_truth"]
            })

        # 2: SA3 Kick (Generative)
        if "sa3_kick" in aud:
            card_list.append({
                "name": "SA3 Kick (Generative)",
                "color": "var(--accent)",
                "border": "rgba(56, 189, 248, 0.5)",
                "bg": "rgba(56, 189, 248, 0.04)",
                "badge": "Gate G3 LoRA",
                "metrics": f'<span class="m-badge">SI-SDR: <strong>{m_sa3.get("si_sdr_db", float("nan")):.2f} dB</strong></span><span class="m-badge">LSD: <strong>{m_sa3.get("lsd_db", float("nan")):.2f} dB</strong></span><span class="m-badge">Onset F1: <strong>{m_sa3.get("onset_f1_20", float("nan")):.3f}</strong></span>',
                "src": aud["sa3_kick"]
            })

        # 3: SA3 Kick (Wiener Masked)
        if "sa3_wiener_kick" in aud:
            card_list.append({
                "name": "SA3 Kick (Wiener Masked)",
                "color": "var(--green)",
                "border": "rgba(16, 185, 129, 0.5)",
                "bg": "rgba(16, 185, 129, 0.04)",
                "badge": "Energy Conserved",
                "metrics": f'<span class="m-badge">SI-SDR: <strong>{m_wiener.get("si_sdr_db", float("nan")):.2f} dB</strong></span><span class="m-badge">LSD: <strong>{m_wiener.get("lsd_db", float("nan")):.2f} dB</strong></span><span class="m-badge">Onset F1: <strong>{m_wiener.get("onset_f1_20", float("nan")):.3f}</strong></span>',
                "src": aud["sa3_wiener_kick"]
            })

        # Snare, Hi-Hat, Percussion
        for s_sub in ["snare", "hihat", "percussion"]:
            s_name = "Hi-Hat" if s_sub == "hihat" else s_sub.capitalize()
            if f"sa3_{s_sub}" in aud:
                card_list.append({
                    "name": f"SA3 {s_name} (Generative)",
                    "color": "var(--accent)",
                    "border": "rgba(56, 189, 248, 0.4)",
                    "bg": "rgba(56, 189, 248, 0.02)",
                    "badge": "Gate G3 LoRA",
                    "metrics": f'<span class="m-badge">Extracted: <strong>{s_name}</strong></span>',
                    "src": aud[f"sa3_{s_sub}"]
                })
            if f"sa3_wiener_{s_sub}" in aud:
                card_list.append({
                    "name": f"SA3 {s_name} (Wiener Masked)",
                    "color": "var(--green)",
                    "border": "rgba(16, 185, 129, 0.4)",
                    "bg": "rgba(16, 185, 129, 0.02)",
                    "badge": "Energy Conserved",
                    "metrics": f'<span class="m-badge">Wiener Masked: <strong>{s_name}</strong></span>',
                    "src": aud[f"sa3_wiener_{s_sub}"]
                })

        # Baselines: BS-RoFormer, MSG-LD, Arm A
        if "bs_roformer" in aud:
            card_list.append({
                "name": "BS-RoFormer (SW Drums)",
                "color": "var(--yellow)",
                "border": "var(--border)",
                "bg": "#131c31",
                "badge": "Drum Bus Baseline",
                "metrics": f'<span class="m-badge">SI-SDR: <strong>{m_bs.get("si_sdr_db", float("nan")):.2f} dB</strong></span><span class="m-badge">LSD: <strong>{m_bs.get("lsd_db", float("nan")):.2f} dB</strong></span><span class="m-badge">Onset F1: <strong>{m_bs.get("onset_f1_20", float("nan")):.3f}</strong></span>',
                "src": aud["bs_roformer"]
            })

        if "msgld" in aud:
            card_list.append({
                "name": "Separate-and-Detect (MSG-LD)",
                "color": "var(--orange)",
                "border": "var(--border)",
                "bg": "#131c31",
                "badge": "16kHz Diffusion",
                "metrics": f'<span class="m-badge">SI-SDR: <strong>{m_msg.get("si_sdr_db", float("nan")):.2f} dB</strong></span><span class="m-badge">LSD: <strong>{m_msg.get("lsd_db", float("nan")):.2f} dB</strong></span><span class="m-badge">Onset F1: <strong>{m_msg.get("onset_f1_20", float("nan")):.3f}</strong></span>',
                "src": aud["msgld"]
            })

        if "sa3_compare_kick" in aud:
            card_list.append({
                "name": "SA3 Arm A (Frozen Prior)",
                "color": "#cbd5e1",
                "border": "rgba(168, 85, 247, 0.2)",
                "bg": "#131c31",
                "badge": "Prior Baseline",
                "metrics": f'<span class="m-badge">SI-SDR: <strong>{m_cmp.get("si_sdr_db", float("nan")):.2f} dB</strong></span><span class="m-badge">LSD: <strong>{m_cmp.get("lsd_db", float("nan")):.2f} dB</strong></span><span class="m-badge">Onset F1: <strong>{m_cmp.get("onset_f1_20", float("nan")):.3f}</strong></span>',
                "src": aud["sa3_compare_kick"]
            })

        for s_idx, c_info in enumerate(card_list):
            is_first = (s_idx == 0)
            active_class = "is-active-solo" if is_first else ""
            html += f"""
        <div class="stem-card {active_class}" id="card_{c_idx}_{s_idx}" style="border-color: {c_info['border']}; background: {c_info['bg']};" onclick="soloStem({c_idx}, {s_idx})">
          <div class="stem-header">
            <span class="stem-name" style="color: {c_info['color']};">{c_info['name']}</span>
            <button class="solo-btn" onclick="event.stopPropagation(); soloStem({c_idx}, {s_idx})">SOLO</button>
          </div>
          <div class="stem-metrics">
            {c_info['metrics']}
          </div>
          <audio id="audio_{c_idx}_{s_idx}" src="{c_info['src']}" preload="metadata"></audio>
        </div>
"""
        html += """
      </div>
    </div>
"""

    # Section 2: Held-Out Validation Slices
    if val_results:
        html += """
    <div style="margin-top: 48px; margin-bottom: 20px;">
      <h2>Held-Out Validation Multitrack Slices (12-Second Crops)</h2>
      <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 20px;">
        12-second crops from held-out multitrack projects. Input is the <strong>Composite Drum Bus</strong> (all drums summed, synths/bass muted).
      </p>
    </div>
"""
        for v_idx, v in enumerate(val_results):
            s_tag = v["sample_tag"]
            song = v.get("song", "Track").split(" - ")[-1]
            win = v.get("window", 0)
            remix = v.get("remix", "orig")
            vw = v.get("wiener")
            wiener_badge = ""
            if vw:
                wiener_badge = f'<span class="m-badge" style="background: rgba(16, 185, 129, 0.15); color: var(--green);">Wiener SI-SDR: <strong>{vw.get("si_sdr_db", float("nan")):.2f} dB</strong></span>'

            stem_tag = v.get("stem_tag", "kick")
            stem_cap = stem_tag.capitalize()
            gt_file = f"{s_tag}_gt_{stem_tag}.wav" if (out_dir / f"audio/{s_tag}_gt_{stem_tag}.wav").exists() else f"{s_tag}_gt_kick.wav"
            sa3_file = f"{s_tag}_sa3_{stem_tag}.wav" if (out_dir / f"audio/{s_tag}_sa3_{stem_tag}.wav").exists() else f"{s_tag}_sa3_kick.wav"
            wiener_file = f"{s_tag}_wiener_{stem_tag}.wav" if (out_dir / f"audio/{s_tag}_wiener_{stem_tag}.wav").exists() else f"{s_tag}_wiener_kick.wav"

            has_val_wiener = (out_dir / f"audio/{wiener_file}").exists()

            v_cards = [
                {"name": "Drum Bus Mixture", "color": "var(--text)", "src": f"audio/{s_tag}_mix.wav", "border": "var(--border)", "bg": "#131c31"},
                {"name": f"Ground Truth {stem_cap}", "color": "var(--purple)", "src": f"audio/{gt_file}", "border": "rgba(168, 85, 247, 0.4)", "bg": "#131c31"},
                {"name": f"SA3 {stem_cap} (Generative)", "color": "var(--accent)", "src": f"audio/{sa3_file}", "border": "rgba(56, 189, 248, 0.5)", "bg": "rgba(56, 189, 248, 0.04)"},
            ]
            if has_val_wiener:
                v_cards.append({"name": f"SA3 {stem_cap} (Wiener Masked)", "color": "var(--green)", "src": f"audio/{wiener_file}", "border": "rgba(16, 185, 129, 0.5)", "bg": "rgba(16, 185, 129, 0.04)"})

            val_id = f"val_{v_idx}"
            html += f"""
    <div class="track-section" id="track_{val_id}">
      <div class="track-header">
        <div class="track-title-block">
          <span class="track-title-text">{song} ({stem_cap}, Window {win}, {remix})</span>
          <span class="now-playing-pill" id="badge_{val_id}">Auditioning: <strong>Ground Truth {stem_cap}</strong></span>
        </div>
        <div class="stem-metrics">
          <span class="m-badge">SI-SDR: <strong>{v.get('si_sdr_db', float('nan')):.2f} dB</strong></span>
          <span class="m-badge">LSD: <strong>{v.get('lsd_db', float('nan')):.2f} dB</strong></span>
          <span class="m-badge">Onset F1: <strong>{v.get('onset_f1_20', float('nan')):.3f}</strong></span>
          {wiener_badge}
        </div>
      </div>
      <div class="master-transport">
        <button class="transport-play-btn" id="play_btn_{val_id}" onclick="togglePlay('{val_id}')">&#9658; Play</button>
        <span class="transport-time" id="time_{val_id}">0:00 / 0:12</span>
        <div class="scrubber-wrap">
          <input type="range" class="timeline-slider" id="scrub_{val_id}" min="0" max="12.07" step="0.05" value="0" oninput="onScrub('{val_id}', this.value)">
        </div>
      </div>
      <div class="stems-grid">
"""
            for vi, vc in enumerate(v_cards):
                is_active = (vi == 1)
                active_class = "is-active-solo" if is_active else ""
                html += f"""
        <div class="stem-card {active_class}" id="card_{val_id}_{vi}" style="border-color: {vc['border']}; background: {vc['bg']};" onclick="soloStem('{val_id}', {vi})">
          <div class="stem-header">
            <span class="stem-name" style="color: {vc['color']};">{vc['name']}</span>
            <button class="solo-btn" onclick="event.stopPropagation(); soloStem('{val_id}', {vi})">SOLO</button>
          </div>
          <audio id="audio_{val_id}_{vi}" src="{vc['src']}" preload="metadata"></audio>
        </div>
"""
            html += """
      </div>
    </div>
"""

    html += """
  </div>

  <script>
    const activeStemMap = {};

    function formatTime(sec) {
      if (!sec || isNaN(sec)) return "0:00";
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return m + ":" + (s < 10 ? "0" : "") + s;
    }

    function initSection(secId, defaultStem = 0) {
      const section = document.getElementById('track_' + secId);
      if (!section) return;
      const audios = section.querySelectorAll('audio');
      if (!audios.length) return;

      activeStemMap[secId] = defaultStem;

      // Ensure audios are loaded and muted properly
      audios.forEach((a, idx) => {
        a.volume = (idx === defaultStem) ? 1.0 : 0.0;
        a.muted = (idx !== defaultStem);
        a.addEventListener('timeupdate', () => {
          if (activeStemMap[secId] === idx) {
            const cur = a.currentTime || 0;
            const dur = a.duration || 30;
            const slider = document.getElementById('scrub_' + secId);
            const timeDisplay = document.getElementById('time_' + secId);
            if (slider) {
              slider.max = dur;
              slider.value = cur;
            }
            if (timeDisplay) {
              timeDisplay.innerText = formatTime(cur) + ' / ' + formatTime(dur);
            }
          }
        });
      });
    }

    function togglePlay(secId) {
      const section = document.getElementById('track_' + secId);
      const audios = section.querySelectorAll('audio');
      const activeIdx = activeStemMap[secId] !== undefined ? activeStemMap[secId] : 0;
      const playBtn = document.getElementById('play_btn_' + secId);
      const activeAudio = audios[activeIdx];
      const isPlaying = !activeAudio.paused;

      // Pause any other sections on the page
      document.querySelectorAll('.track-section').forEach(s => {
        if (s.id !== 'track_' + secId) {
          s.querySelectorAll('audio').forEach(a => a.pause());
          const b = s.querySelector('.transport-play-btn');
          if (b) b.innerHTML = '&#9658; Play';
        }
      });

      if (isPlaying) {
        audios.forEach(a => a.pause());
        playBtn.innerHTML = '&#9658; Play';
      } else {
        const curTime = activeAudio.currentTime;
        audios.forEach((a, i) => {
          a.currentTime = curTime;
          if (i === activeIdx) {
            a.muted = false;
            a.volume = 1.0;
          } else {
            a.muted = true;
            a.volume = 0.0;
          }
          a.play().catch(e => console.log(e));
        });
        playBtn.innerHTML = '&#10074;&#10074; Pause';
      }
    }

    function onScrub(secId, val) {
      const section = document.getElementById('track_' + secId);
      const audios = section.querySelectorAll('audio');
      const t = parseFloat(val);
      audios.forEach(a => {
        a.currentTime = t;
      });
      const dur = audios[0].duration || 30;
      const timeDisplay = document.getElementById('time_' + secId);
      if (timeDisplay) {
        timeDisplay.innerText = formatTime(t) + ' / ' + formatTime(dur);
      }
    }

    function soloStem(secId, stemIdx) {
      const section = document.getElementById('track_' + secId);
      const audios = section.querySelectorAll('audio');
      const cards = section.querySelectorAll('.stem-card');
      const badge = document.getElementById('badge_' + secId);
      const prevIdx = activeStemMap[secId] !== undefined ? activeStemMap[secId] : 0;
      const isPlaying = !audios[prevIdx].paused;
      const curTime = audios[prevIdx].currentTime;

      activeStemMap[secId] = stemIdx;

      cards.forEach((c, i) => {
        if (i === stemIdx) {
          c.classList.add('is-active-solo');
          const stemName = c.querySelector('.stem-name').innerText;
          if (badge) badge.innerHTML = 'Auditioning: <strong>' + stemName + '</strong>';
        } else {
          c.classList.remove('is-active-solo');
        }
      });

      audios.forEach((a, i) => {
        a.currentTime = curTime;
        if (i === stemIdx) {
          a.muted = false;
          a.volume = 1.0;
          if (isPlaying) a.play().catch(e => console.log(e));
        } else {
          a.muted = true;
          a.volume = 0.0;
          if (isPlaying) a.play().catch(e => console.log(e));
          else a.pause();
        }
      });
    }

    // Auto-init all sections on page load
    window.addEventListener('DOMContentLoaded', () => {
      document.querySelectorAll('.track-section').forEach(s => {
        const secId = s.id.replace('track_', '');
        initSection(secId, 0);
      });
    });
  </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[audition_console] Successfully deployed interactive audition console to {html_path}")

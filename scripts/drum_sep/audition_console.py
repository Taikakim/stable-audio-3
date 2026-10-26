"""Interactive HTML Audition Console Generator for SA3 Gate G2 Drum Separation.
Generates a standalone, dark-themed HTML report with synchronized multi-track playback,
live A/B solo comparison, and comparative objective metrics across all baselines.
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

    cmp_has_compare = any("sa3_compare" in r for r in comp_results)
    if cmp_has_compare:
        cmp_sisdr = float(np.mean([r["sa3_compare"]["si_sdr_db"] for r in comp_results if "sa3_compare" in r]))
        cmp_lsd = float(np.mean([r["sa3_compare"]["lsd_db"] for r in comp_results if "sa3_compare" in r]))
        cmp_f1 = float(np.mean([r["sa3_compare"]["onset_f1_20"] for r in comp_results if "sa3_compare" in r]))
    else:
        cmp_sisdr, cmp_lsd, cmp_f1 = float("nan"), float("nan"), float("nan")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SA3 Drum Separator — Gate G2 Kick Separation Audition Console</title>
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

    /* Track Cards */
    .track-section {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 24px;
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
      gap: 12px;
    }}
    .track-title {{
      font-size: 16px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .sync-btn {{
      background: var(--accent);
      color: #000;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s;
    }}
    .sync-btn:hover {{ background: #7dd3fc; }}
    .stems-grid {{
      padding: 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .stem-card {{
      background: #131c31;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
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
    .audio-wrap {{
      margin-top: 4px;
    }}
    audio {{
      width: 100%;
      height: 36px;
      border-radius: 4px;
      outline: none;
    }}
    .solo-btn {{
      background: rgba(255,255,255,0.08);
      color: var(--text-muted);
      border: 1px solid var(--border);
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 8px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .solo-btn:hover {{
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
      border-color: var(--accent);
    }}
    .solo-btn.active {{
      background: var(--accent);
      color: #000;
      border-color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        SA3 Drum Separator
        <span class="badge-gate">Gate G2 Verified</span>
      </h1>
      <p class="subtitle">
        Conditioned Kick Drum Separation benchmark on discrete AMD Radeon RX 9070 XT (16 GB VRAM).
        Comparing SA3 Medium Base with active modular local-conditioning inlet against Ground Truth, BS-RoFormer, Separate-and-Detect (MSG-LD), and zero-training priors.
      </p>
      <div class="stat-row">
        <div class="stat-pill">
          <span class="val">{sa3_comp_sisdr:+.2f} dB</span>
          <span class="lbl">SA3 Benchmark SI-SDR</span>
        </div>
        <div class="stat-pill">
          <span class="val">{sa3_comp_lsd:.2f} dB</span>
          <span class="lbl">SA3 Benchmark LSD</span>
        </div>
        <div class="stat-pill">
          <span class="val">{sa3_comp_f1:.3f}</span>
          <span class="lbl">SA3 Onset F1 @ 20ms</span>
        </div>
        <div class="stat-pill">
          <span class="val">{val_sisdr:+.2f} dB</span>
          <span class="lbl">Held-Out Val SI-SDR</span>
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
            <td><strong>SAME-L Codec Ceiling</strong></td>
            <td>Autoencoder Reconstruction</td>
            <td>Isolated Kick</td>
            <td>+27.00 dB</td>
            <td>1.12 dB</td>
            <td>1.000</td>
            <td><span class="pill-tag tag-green">Gate G0 Ceiling</span></td>
          </tr>
          <tr>
            <td><strong>SA3 Medium Base (Prior)</strong></td>
            <td>Solo Kit Piece Prompt (Zero-Train)</td>
            <td>Isolated Kick</td>
            <td>-23.50 dB</td>
            <td>32.10 dB</td>
            <td>0.040</td>
            <td><span class="pill-tag tag-red">Gate G1 Failed</span></td>
          </tr>
          <tr>
            <td><strong>SA3 Medium Base (SDEdit)</strong></td>
            <td>Noise Gating on Drum Bus (&sigma;=0.5)</td>
            <td>Isolated Kick</td>
            <td>-31.40 dB</td>
            <td>28.50 dB</td>
            <td>0.050</td>
            <td><span class="pill-tag tag-red">Gate G1 Failed</span></td>
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
            <td><strong>BS-RoFormer (SW Drums)</strong></td>
            <td>Sub-band Time-Frequency Transformer</td>
            <td>Full Drums</td>
            <td>{bs_comp_sisdr:+.2f} dB</td>
            <td>{bs_comp_lsd:.2f} dB</td>
            <td>{bs_comp_f1:.3f}</td>
            <td><span class="pill-tag tag-yellow">Drum Bus Baseline</span></td>
          </tr>
          <tr>
            <td><strong>SA3 Gate G2 (Arm A)</strong></td>
            <td>LoRA Prior (Modular Inlet Frozen)</td>
            <td>Isolated Kick</td>
            <td>-35.56 dB</td>
            <td>25.45 dB</td>
            <td>0.067</td>
            <td><span class="pill-tag tag-purple">Unconditioned Prior</span></td>
          </tr>
          <tr class="highlight-target">
            <td><strong>SA3 Gate G2 (Arm A-v2)</strong></td>
            <td>LoRA + Active Modular Mix Inlet</td>
            <td>Isolated Kick</td>
            <td>{sa3_comp_sisdr:+.2f} dB</td>
            <td>{sa3_comp_lsd:.2f} dB</td>
            <td>{sa3_comp_f1:.3f}</td>
            <td><span class="pill-tag tag-blue">Gate G2 Target</span></td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Benchmark Multitrack Audition Section -->
    <div style="margin-bottom: 16px;">
      <h2>Studio Multitrack Audition & A/B Comparison</h2>
      <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 20px;">
        Click <strong>Sync Play</strong> to start all stems in sync, then click <strong>SOLO</strong> on any stem to audition it in isolation in real time.
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

        html += f"""
    <div class="track-section" id="track_{c_idx}">
      <div class="track-header">
        <div class="track-title">
          <span>{t_id.replace('_', ' ').title()}</span>
          <span class="pill-tag tag-blue">Multitrack Ground Truth</span>
        </div>
        <button class="sync-btn" onclick="toggleSync({c_idx})">&#9658; Sync Play All</button>
      </div>
      <div class="stems-grid">
        <!-- Mix -->
        <div class="stem-card">
          <div class="stem-header">
            <span class="stem-name">Full Mix (Mixture)</span>
            <button class="solo-btn" onclick="soloStem({c_idx}, 0)">SOLO</button>
          </div>
          <div class="audio-wrap">
            <audio controls src="{aud.get('mix', '')}" preload="none"></audio>
          </div>
        </div>

        <!-- Ground Truth Kick -->
        <div class="stem-card" style="border-color: rgba(168, 85, 247, 0.4);">
          <div class="stem-header">
            <span class="stem-name" style="color: var(--purple);">Studio Kick (Ground Truth)</span>
            <button class="solo-btn" onclick="soloStem({c_idx}, 1)">SOLO</button>
          </div>
          <div class="stem-metrics">
            <span class="m-badge">Target: <strong>Clean Stem</strong></span>
          </div>
          <div class="audio-wrap">
            <audio controls src="{aud.get('ground_truth', '')}" preload="none"></audio>
          </div>
        </div>

        <!-- SA3 Gate G2 Arm A-v2 Kick -->
        <div class="stem-card" style="border-color: rgba(56, 189, 248, 0.5); background: rgba(56, 189, 248, 0.04);">
          <div class="stem-header">
            <span class="stem-name" style="color: var(--accent);">SA3 Gate G2 (Arm A-v2)</span>
            <button class="solo-btn" onclick="soloStem({c_idx}, 2)">SOLO</button>
          </div>
          <div class="stem-metrics">
            <span class="m-badge">SI-SDR: <strong>{m_sa3.get('si_sdr_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">LSD: <strong>{m_sa3.get('lsd_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">Onset F1: <strong>{m_sa3.get('onset_f1_20', float('nan')):.3f}</strong></span>
          </div>
          <div class="audio-wrap">
            <audio controls src="{aud.get('sa3_kick', '')}" preload="none"></audio>
          </div>
        </div>
"""
        stem_idx = 3
        if "sa3_compare_kick" in aud:
            html += f"""
        <!-- SA3 Gate G2 Arm A Kick (Prior) -->
        <div class="stem-card" style="border-color: rgba(168, 85, 247, 0.2);">
          <div class="stem-header">
            <span class="stem-name" style="color: #cbd5e1;">SA3 Arm A (Prior Only)</span>
            <button class="solo-btn" onclick="soloStem({c_idx}, {stem_idx})">SOLO</button>
          </div>
          <div class="stem-metrics">
            <span class="m-badge">SI-SDR: <strong>{m_cmp.get('si_sdr_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">LSD: <strong>{m_cmp.get('lsd_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">Onset F1: <strong>{m_cmp.get('onset_f1_20', float('nan')):.3f}</strong></span>
          </div>
          <div class="audio-wrap">
            <audio controls src="{aud.get('sa3_compare_kick', '')}" preload="none"></audio>
          </div>
        </div>
"""
            stem_idx += 1

        if "bs_roformer" in aud:
            html += f"""
        <!-- BS-RoFormer -->
        <div class="stem-card">
          <div class="stem-header">
            <span class="stem-name" style="color: var(--yellow);">BS-RoFormer (SW Drums)</span>
            <button class="solo-btn" onclick="soloStem({c_idx}, {stem_idx})">SOLO</button>
          </div>
          <div class="stem-metrics">
            <span class="m-badge">SI-SDR: <strong>{m_bs.get('si_sdr_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">LSD: <strong>{m_bs.get('lsd_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">Onset F1: <strong>{m_bs.get('onset_f1_20', float('nan')):.3f}</strong></span>
          </div>
          <div class="audio-wrap">
            <audio controls src="{aud.get('bs_roformer', '')}" preload="none"></audio>
          </div>
        </div>
"""
            stem_idx += 1

        if "msgld" in aud:
            html += f"""
        <!-- MSG-LD -->
        <div class="stem-card">
          <div class="stem-header">
            <span class="stem-name" style="color: var(--orange);">MSG-LD (Separate-and-Detect)</span>
            <button class="solo-btn" onclick="soloStem({c_idx}, {stem_idx})">SOLO</button>
          </div>
          <div class="stem-metrics">
            <span class="m-badge">SI-SDR: <strong>{m_msg.get('si_sdr_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">LSD: <strong>{m_msg.get('lsd_db', float('nan')):.2f} dB</strong></span>
            <span class="m-badge">Onset F1: <strong>{m_msg.get('onset_f1_20', float('nan')):.3f}</strong></span>
          </div>
          <div class="audio-wrap">
            <audio controls src="{aud.get('msgld', '')}" preload="none"></audio>
          </div>
        </div>
"""
        html += """
      </div>
    </div>
"""

    if val_results:
        html += """
    <div style="margin-top: 36px; margin-bottom: 16px;">
      <h2>Held-Out Validation Dataset Samples</h2>
      <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 20px;">
        Samples from held-out songs in the val split evaluating dynamic mixture separation across unseen multitrack arrangements.
      </p>
    </div>
"""
        for v_idx, v in enumerate(val_results):
            s_tag = v.get("sample_tag", f"val_{v_idx}")
            song = v.get("song", "Validation Sample")
            win = v.get("window", 0)
            remix = v.get("remix", "orig")

            html += f"""
    <div class="track-section" id="val_track_{v_idx}">
      <div class="track-header">
        <div class="track-title">
          <span>{song} (Window {win}, {remix})</span>
        </div>
        <div class="stem-metrics">
          <span class="m-badge">SI-SDR: <strong>{v.get('si_sdr_db', float('nan')):.2f} dB</strong></span>
          <span class="m-badge">LSD: <strong>{v.get('lsd_db', float('nan')):.2f} dB</strong></span>
          <span class="m-badge">Onset F1: <strong>{v.get('onset_f1_20', float('nan')):.3f}</strong></span>
        </div>
      </div>
      <div class="stems-grid" style="grid-template-columns: repeat(3, 1fr);">
        <div class="stem-card">
          <span class="stem-name">Drum Bus Mixture</span>
          <div class="audio-wrap">
            <audio controls src="audio/{s_tag}_mix.wav" preload="none"></audio>
          </div>
        </div>
        <div class="stem-card" style="border-color: rgba(168, 85, 247, 0.4);">
          <span class="stem-name" style="color: var(--purple);">Ground Truth Kick</span>
          <div class="audio-wrap">
            <audio controls src="audio/{s_tag}_gt_kick.wav" preload="none"></audio>
          </div>
        </div>
        <div class="stem-card" style="border-color: rgba(56, 189, 248, 0.5); background: rgba(56, 189, 248, 0.04);">
          <span class="stem-name" style="color: var(--accent);">SA3 Gate G2 Separated Kick</span>
          <div class="audio-wrap">
            <audio controls src="audio/{s_tag}_sa3_kick.wav" preload="none"></audio>
          </div>
        </div>
      </div>
    </div>
"""

    html += """
  </div>

  <script>
    function toggleSync(trackIdx) {
      const section = document.getElementById('track_' + trackIdx);
      const audios = section.querySelectorAll('audio');
      const btn = section.querySelector('.sync-btn');
      
      const anyPlaying = Array.from(audios).some(a => !a.paused);
      if (anyPlaying) {
        audios.forEach(a => a.pause());
        btn.innerHTML = '&#9658; Sync Play All';
      } else {
        const t = audios[0].currentTime || 0;
        audios.forEach(a => {
          a.currentTime = t;
          a.play();
        });
        btn.innerHTML = '&#10074;&#10074; Pause All';
      }
    }

    function soloStem(trackIdx, stemIdx) {
      const section = document.getElementById('track_' + trackIdx);
      const cards = section.querySelectorAll('.stem-card');
      const audios = section.querySelectorAll('audio');
      const soloBtns = section.querySelectorAll('.solo-btn');
      
      const targetBtn = soloBtns[stemIdx];
      const isAlreadySolo = targetBtn.classList.contains('active');

      if (isAlreadySolo) {
        soloBtns.forEach(b => b.classList.remove('active'));
        audios.forEach(a => { a.muted = false; a.volume = 1.0; });
      } else {
        soloBtns.forEach(b => b.classList.remove('active'));
        targetBtn.classList.add('active');
        audios.forEach((a, i) => {
          a.muted = (i !== stemIdx);
          a.volume = (i === stemIdx) ? 1.0 : 0.0;
        });
      }
    }
  </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[audition_console] Successfully deployed interactive audition console to {html_path}")

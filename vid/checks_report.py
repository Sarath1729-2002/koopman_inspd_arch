"""One page with every pre-registered check from Nov 24 -> CHECKS.md (reads whatever result files exist)."""
import json, os, re

L = ['# Pre-registered checks (Nov 24)', '', '| check | question | result | key numbers |', '|---|---|---|---|']


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def row(k, q, ok, txt):
    L.append(f"| {k} | {q} | {'—' if ok is None else ('PASS' if ok else 'FAIL')} | {txt} |")


for root, keys in (('results_sf_long/SF_CAMPAIGN.md', ('V1', 'V2', 'V3')), ('results_sf/SF_CAMPAIGN.md', ('E1',))):
    if os.path.exists(root):
        t = open(root).read()
        for k in keys:
            m = re.search(rf'- {k}: (PASS|FAIL)', t)
            if m:
                q = {'V1': 'CausVid 4 min: re-encode helps (prompts 0-7)', 'V2': 'CausVid 4 min: re-encode helps on FRESH prompts 8-23',
                     'E1': 'CausVid 60 s: drift-triggered re-encode helps on FRESH prompts 16-31',
                     'V3': 'CausVid 4 min: re-encode helps on FRESH prompts 24-39'}[k]
                row(k, q, m.group(1) == 'PASS', f'see {root}')
g = j('results_growth.json')
if g:
    ob = g['G1'].get('oasis_base')
    row('G1', 'Oasis drift grows exponentially without correction', g.get('G1_pass'),
        f"winners {ob['winner_fraction']}, rate {ob['exp_rate_median'][0]:+.4f}/frame, doubling {ob['doubling_time']}" if ob else 'no Oasis data')
    for k, s in g['G1'].items():
        if k != 'oasis_base':
            row('G1 (info)', k, None, f"winners {s['winner_fraction']}, last/floor {s['drift_last_over_floor']:.2f}")
    if 'G2_pass' in g:
        row('G2', 'regrowth after a correction predicts the best period', g['G2_pass'],
            f"P_pred {g['G2'].get('P_pred')} vs best {g['G2'].get('P_best')} (FD {g['G2'].get('fd_by_period')})")
if g and 'G3' in g:
    row('G3', 'a linear error model forecasts where Oasis drift settles', g.get('G3_pass'),
        f"lam {g['G3']['lam_median']:.3f}, within 25% in {g['G3']['frac_within_25pct']:.0%}")
for root, keys in (('results_sf_long/SF_CAMPAIGN.md', ('S4', 'B1')),):
    if os.path.exists(root):
        t = open(root).read()
        for k in keys:
            m_ = re.search(rf'- {k}: (PASS|FAIL)', t)
            if m_:
                row(k, {'S4': 'Self-Forcing 4 min: re-encode helps (drift 4.7x)', 'B1': 'band-limited beats plain re-encode'}[k],
                    m_.group(1) == 'PASS', f'see {root}')
mj = j('results_sf_long/mech.json')
if mj and 'M1' in mj:
    row('M1', 'CausVid: some image statistic drifts and re-encoding holds it back', mj['M1'], ', '.join(mj['M1_stats']) or 'none')
for arm in ('base', 'sink4'):
    k = j(f'results_koop/koopman_{arm}.json')
    if k:
        row('K1' + ('' if arm == 'base' else ' (sink4)'), f'Koopman residual tracks re-encode drift ({arm})', k['K1'],
            f"Spearman {k['spearman_koopman_vs_reencode']:+.2f}, late/window {k['late_over_window']:.2f}")
        row('K2' + ('' if arm == 'base' else ' (sink4)'), f'rho(K) predicts the growth law ({arm})', k['K2'],
            f"rho {k['rho']:.3f} -> {k['law_predicted']}; observed {k['law_observed']}")
for sfx, lab in (('', '1 epoch/stage'), ('_e3', '3 epochs/stage')):
    c = j(f'results_rdiag/coconut{sfx}.json')
    if c:
        r = c['runs']['none_10']
        row('C1', f'Coconut drift large and informative ({lab})', c['C1'], f"drift/floor {r['drift_over_floor_last']:.2f}, AUROC {r['auroc_wrong']:.2f}")
        row('C2', f'snapping thoughts helps ({lab})', c['C2'], 'see coconut json')
        if 'C3' in c:
            row('C3', f'Coconut replicates (acc >= 0.30) ({lab})', c['C3'], f"acc {r['acc']:.3f}")
    d = j(f'results_rdiag/cdir{sfx}.json')
    if d:
        r = d['runs']['10']
        row('D1', f'Coconut drift has an error DIRECTION ({lab})', d['D1'],
            f"AUROC direction {r['auroc_dir']:.3f} vs size {r['auroc_drift_size']:.3f} (raw-thought dir {r['auroc_raw_dir']:.3f})")
        row('D2', f'steering against it helps ({lab})', d['D2'],
            '; '.join(f"n={n}: {d['runs'][n]['steer']['delta_vs_none'][0]:+.3f} vs none, {d['runs'][n]['steer']['delta_vs_random'][0]:+.3f} vs random" for n in d['runs']))
o = j('results_rdiag/odir.json')
if o:
    row('OD1', 'Ouro drift has a damage DIRECTION', o['OD1'], f"corr direction {o['OD1_detail']['corr_dir']:+.3f} vs size {o['OD1_detail']['corr_size']:+.3f}")
    row('OD2', 'steering against it lowers depth-8 loss', o['OD2'],
        f"vs none {o['OD2_detail']['diff_vs_none'][0]:+.4f}, vs random {o['OD2_detail']['diff_vs_random'][0]:+.4f}")
    h = o['OH1_detail']
    row('OH1', 'drift as a stopping signal beats fixed 4 loops', o['OH1'],
        f"NLL stop {h['nll_stop'][0]:.4f} vs fixed-4 {h['nll_fixed4'][0]:.4f} vs oracle {h['nll_oracle'][0]:.4f}")
for tag in ('', '_26'):
    o2 = j(f'results_rdiag/odir2{tag}.json')
    if not o2:
        continue
    lab = 'Ouro-2.6B' if tag else 'Ouro-1.4B'
    q = {'N1': 'u beats 50 random + 20 shuffled-label directions', 'N2': 'u beats label-free mean drift direction',
         'N3': 'off-manifold u beats raw-state direction', 'U1': 'steered loop 5/6 beats plain loop 4',
         'U2': 'steering 6 loops raises GSM8k accuracy', 'KP1': 'Koopman direction aligns with u (|cos| >= 0.5)',
         'KP2': 'label-free Koopman direction steers', 'A1': 'mode-seeking attractor (step shrinks, confident + wrong)',
         'A2': 'u beats entropy drop as a damage signal', 'OH2': 'direction-score stopping beats fixed 4',
         'G1o': 'u transfers to other text'}
    det = {'N1': lambda: f"beats {o2['nulls']['u_better_than_random_frac']:.0%} random, {o2['nulls']['u_better_than_shuffled_frac']:.0%} shuffled",
           'N2': lambda: f"u - mean {o2['N2_detail'][0]:+.4f}", 'N3': lambda: f"u - raw {o2['N3_detail'][0]:+.4f}",
           'U1': lambda: '; '.join(f"{k}: {v['steered'][0]:.4f} vs d4 {v['unsteered_depth4'][0]:.4f}" for k, v in o2['U1_detail'].items()),
           'U2': lambda: '; '.join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
                          for k, v in o2['U2_detail'].items() if isinstance(v, (float, list))),
           'KP1': lambda: f"cos kmode {o2['cos_with_u']['kmode']:+.2f}, katt {o2['cos_with_u']['katt']:+.2f}, rho {o2['koopman']['rho']:.3f}",
           'KP2': lambda: f"kmode {o2['steer_vs_none']['kmode'][0]:+.4f}, katt {o2['steer_vs_none']['katt'][0]:+.4f} vs none",
           'A1': lambda: f"step Spearman {o2['step_spearman_beyond4']:+.2f}, entropy d4/d8 {o2['dynamics']['4']['entropy']:.3f}/{o2['dynamics']['8']['entropy']:.3f}" if '4' in o2['dynamics'] else '',
           'A2': lambda: f"dir {o2['A2_detail']['corr_dir']:+.3f} vs entropy {o2['A2_detail']['corr_entropy_drop']:+.3f}",
           'OH2': lambda: f"{o2['OH2_detail']['nll_stop'][0]:.4f} vs {o2['OH2_detail']['nll_fixed4'][0]:.4f}",
           'G1o': lambda: f"vs none {o2['G1o_detail']['u_vs_none'][0]:+.4f}, vs random {o2['G1o_detail']['u_vs_random'][0]:+.4f}"}
    for k in q:
        if k in o2:
            try:
                d_ = det[k]()
            except Exception:
                d_ = ''
            row(k, f'{q[k]} ({lab})', o2[k], d_)
open('CHECKS.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

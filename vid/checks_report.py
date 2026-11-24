"""One page with every pre-registered check from Nov 24 -> CHECKS.md (reads whatever result files exist)."""
import json, os, re

L = ['# Pre-registered checks (Nov 24)', '', '| check | question | result | key numbers |', '|---|---|---|---|']


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def row(k, q, ok, txt):
    L.append(f"| {k} | {q} | {'—' if ok is None else ('PASS' if ok else 'FAIL')} | {txt} |")


for root, keys in (('results_sf_long/SF_CAMPAIGN.md', ('V1', 'V2')), ('results_sf/SF_CAMPAIGN.md', ('E1',))):
    if os.path.exists(root):
        t = open(root).read()
        for k in keys:
            m = re.search(rf'- {k}: (PASS|FAIL)', t)
            if m:
                q = {'V1': 'CausVid 4 min: re-encode helps (prompts 0-7)', 'V2': 'CausVid 4 min: re-encode helps on FRESH prompts 8-23',
                     'E1': 'CausVid 60 s: drift-triggered re-encode helps on FRESH prompts 16-31'}[k]
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
open('CHECKS.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

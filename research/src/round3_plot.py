"""Plot the frozen round-three development summary without rescoring any model."""
from __future__ import annotations
import os
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

SOURCE = ROOT / 'outputs/round3/evaluation/development/results.json'
OUTPUT = ROOT / 'reports/round3_development_stability.png'
CANDIDATES = ['recency_blend', 'regional_trend', 'recency_regional']
LABELS = {'recency_blend': 'Recency blend', 'regional_trend': 'Regional trend',
          'recency_regional': 'Recency + regional'}
COLORS = {'recency_blend': '#C2410C', 'regional_trend': '#0369A1',
          'recency_regional': '#7E22CE'}
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_sha = digest(SOURCE)
    frozen_path = ROOT / 'outputs/round3/frozen_candidates.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8-sig'))
    receipt = frozen_path.with_suffix('.sha256').read_text(encoding='ascii').strip()
    if receipt != digest(frozen_path):
        raise ValueError('Frozen-result receipt mismatch')
    expected = frozen['development_evaluation_artifact_sha256'][SOURCE.relative_to(ROOT).as_posix()]
    if source_sha != expected:
        raise ValueError('Development results differ from the frozen artifact')
    report = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
    if report['split'] != 'development' or report['unit'] != 'TWD':
        raise ValueError('This figure requires frozen development MAE in TWD')
    metrics = report['metrics']
    control = metrics['v2_control']
    overall = [100 * (metrics[name]['overall_mae'] / control['overall_mae'] - 1) for name in CANDIDATES]
    weighted = [100 * (metrics[name]['weighted_mae'] / control['weighted_mae'] - 1) for name in CANDIDATES]
    quarterly = {name: [100 * (metrics[name]['forecast_quarters'][q]['mae'] /
                              control['forecast_quarters'][q]['mae'] - 1) for q in QUARTERS]
                 for name in CANDIDATES}
    if not np.isfinite(np.array(overall + weighted + sum(quarterly.values(), []))).all():
        raise ValueError('Nonfinite relative MAE change')
    if (report['thresholds']['development_min_overall_improvement'] != .01 or
            report['thresholds']['development_min_weighted_improvement'] != .01 or
            report['thresholds']['development_max_quarter_deterioration'] != .05):
        raise ValueError('Frozen acceptance thresholds differ from the plotted reference lines')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 14, 'axes.labelsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#CBD5E1', 'axes.labelcolor': '#334155',
                         'text.color': '#0F172A', 'xtick.color': '#475569',
                         'ytick.color': '#475569', 'axes.unicode_minus': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.2, 6.8), dpi=160)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.135, right=.975, bottom=.28, top=.77, wspace=.29)
    fig.suptitle('Development only: candidate stability vs. v2', fontsize=21,
                 fontweight='bold', x=.055, y=.965, ha='left')
    dates = report['splits']['development']
    fig.text(.055, .893,
             f"Relative MAE change (%)  |  Lower is better  |  {dates['start']} to {dates['end_inclusive']}",
             fontsize=12.5, color='#475569')

    positions = np.arange(len(CANDIDATES))[::-1]
    left_values = np.array(overall + weighted)
    lower, upper = min(-1.0, left_values.min()), max(0.0, left_values.max())
    width = max(upper - lower, 1.0)
    left.set_xlim(lower - .2 * width, upper + .28 * width)
    left.set_ylim(-.5, 2.5)
    for i, name in enumerate(CANDIDATES):
        color, y = COLORS[name], positions[i]
        left.barh(y + .145, overall[i], height=.24, color=color, zorder=3)
        left.barh(y - .145, weighted[i], height=.24, color='white', edgecolor=color,
                  linewidth=1.5, hatch='////', zorder=3)
        for offset, value in ((.145, overall[i]), (-.145, weighted[i])):
            left.annotate(f'{value:+.2f}%', (value, y + offset),
                          xytext=(6 if value >= 0 else -6, 0), textcoords='offset points',
                          ha='left' if value >= 0 else 'right', va='center', fontsize=10.5,
                          color=color, fontweight='medium')
    left.axvline(0, color='#64748B', linewidth=1, zorder=2)
    left.axvline(-1, color='#15803D', linewidth=1.5, linestyle='--', zorder=2)
    left.set_yticks(positions, [LABELS[name] for name in CANDIDATES])
    left.tick_params(axis='y', length=0, pad=10)
    left.set_xlabel('Change from v2 MAE (%)', labelpad=10)
    left.set_title('Overall and weighted MAE', loc='left', pad=18, fontweight='bold')
    left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}%'))
    left.xaxis.set_major_locator(MaxNLocator(nbins=6))
    left.grid(axis='x', color='#E2E8F0', linewidth=.8, zorder=0)
    left.legend(handles=[Patch(facecolor='#64748B', label='Overall MAE'),
                         Patch(facecolor='white', edgecolor='#64748B', hatch='////', label='Test-composition-weighted MAE'),
                         Line2D([0], [0], color='#15803D', linestyle='--', linewidth=1.5, label='-1% required improvement')],
                loc='upper center', bbox_to_anchor=(.5, -.24), frameon=False, fontsize=10.3)

    x = np.arange(4)
    markers = ['o', 's', '^']
    for name, marker in zip(CANDIDATES, markers):
        right.plot(x, quarterly[name], color=COLORS[name], marker=marker,
                   markersize=7, linewidth=2, label=LABELS[name], zorder=3)
    right.axhline(0, color='#64748B', linewidth=1, zorder=2)
    right.axhline(5, color='#B91C1C', linewidth=1.5, linestyle='--', zorder=2)
    right.annotate('+5% quarterly deterioration limit', (0.02, 5),
                   xycoords=('axes fraction', 'data'), xytext=(0, 6), textcoords='offset points',
                   color='#B91C1C', fontsize=10.2, ha='left', va='bottom')
    all_quarters = np.array(list(quarterly.values()))
    right.set_ylim(min(-1.0, all_quarters.min() - .65), max(6.0, all_quarters.max() + .8))
    right.set_xlim(-.15, 3.15)
    right.set_xticks(x, ['Q1\nDays 0-91', 'Q2\nDays 92-183', 'Q3\nDays 184-275', 'Q4\nDays 276+'])
    right.set_ylabel('Change from v2 MAE (%)', labelpad=10)
    right.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}%'))
    right.yaxis.set_major_locator(MaxNLocator(nbins=6))
    right.set_title('Stability across four forecast quarters', loc='left', pad=18, fontweight='bold')
    right.grid(axis='y', color='#E2E8F0', linewidth=.8, zorder=0)
    right.legend(loc='upper center', bbox_to_anchor=(.5, -.24), frameon=False, fontsize=10.3)

    fig.text(.055, .03,
             'Development only. Historical labels were reused; these diagnostics are not an independent or private test.',
             fontsize=10.5, color='#64748B', ha='left')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor='white',
                metadata={'Description': 'Frozen development-only relative MAE stability. Lower is better.',
                          'SourceSHA256': source_sha})
    plt.close(fig)
    if digest(SOURCE) != source_sha:
        raise RuntimeError('Input report changed while plotting')
    print(json.dumps({'output': str(OUTPUT), 'source_sha256': source_sha,
                      'overall_change_pct': dict(zip(CANDIDATES, overall)),
                      'weighted_change_pct': dict(zip(CANDIDATES, weighted)),
                      'quarterly_change_pct': quarterly}, ensure_ascii=True))


if __name__ == '__main__':
    main()

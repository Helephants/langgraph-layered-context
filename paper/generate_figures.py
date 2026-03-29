"""
Generate publication-quality figures for The Silicon Mirror paper.
Produces PDF figures suitable for LaTeX \includegraphics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BLUE   = '#2563EB'
RED    = '#DC2626'
GREEN  = '#16A34A'
ORANGE = '#EA580C'
GRAY   = '#6B7280'
LIGHT_BLUE  = '#DBEAFE'
LIGHT_RED   = '#FEE2E2'
LIGHT_GREEN = '#DCFCE7'
LIGHT_ORANGE = '#FED7AA'


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Pipeline Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════
def fig1_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.5, 3)
    ax.axis('off')

    # Node positions (x, y)
    nodes = {
        'User\nMessage':      (0.5, 1.5),
        'Trait\nClassifier':   (2.7, 1.5),
        'BAC':                 (5.0, 1.5),
        'Generator':           (7.2, 1.5),
        'Critic':              (9.4, 1.5),
        'Response':            (11.5, 1.5),
    }

    # Node styles
    io_style = dict(boxstyle='round,pad=0.4', facecolor='#F3F4F6', edgecolor=GRAY, linewidth=1.2)
    stage_style = dict(boxstyle='round,pad=0.4', facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5)
    critic_style = dict(boxstyle='round,pad=0.4', facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=1.5)

    styles = {
        'User\nMessage': io_style,
        'Trait\nClassifier': stage_style,
        'BAC': stage_style,
        'Generator': stage_style,
        'Critic': critic_style,
        'Response': io_style,
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        style = styles[name]
        bbox = FancyBboxPatch((x - 0.85, y - 0.4), 1.7, 0.8, **style)
        ax.add_patch(bbox)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold',
                color='#1F2937')

    # Arrows (main flow)
    arrow_kw = dict(arrowstyle='->', color=GRAY, lw=1.5,
                    connectionstyle='arc3,rad=0', mutation_scale=15)

    pairs = [
        ('User\nMessage', 'Trait\nClassifier'),
        ('Trait\nClassifier', 'BAC'),
        ('BAC', 'Generator'),
        ('Generator', 'Critic'),
        ('Critic', 'Response'),
    ]
    for a, b in pairs:
        ax1, ay1 = nodes[a]
        ax2, ay2 = nodes[b]
        ax.annotate('', xy=(ax2 - 0.85, ay2), xytext=(ax1 + 0.85, ay1),
                     arrowprops=arrow_kw)

    # Edge labels
    edge_labels = [
        (1.6, 2.05, r'$\mathbf{t}$', 8),
        (3.85, 2.05, 'risk $R$', 7),
        (6.1, 2.05, 'adapter', 7),
        (8.3, 2.05, 'draft', 7),
        (10.45, 2.05, 'PASS', 7),
    ]
    for x, y, txt, fs in edge_labels:
        ax.text(x, y, txt, ha='center', va='bottom', fontsize=fs, color=GRAY,
                fontstyle='italic')

    # Veto loop (dashed red arrow curving below)
    veto_x = [9.4, 9.4, 7.2, 7.2]
    veto_y = [1.1, 0.2, 0.2, 1.1]
    ax.annotate('', xy=(7.2, 1.1), xytext=(9.4, 1.1),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.8,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=-0.5',
                                mutation_scale=15))
    ax.text(8.3, -0.05, 'VETO + friction\n($k \\leq 2$ rewrites)', ha='center',
            va='top', fontsize=7.5, color=RED, fontstyle='italic')

    # Sub-annotations
    ax.text(2.7, 0.85, r'$\alpha, \sigma, \gamma, \tau$', ha='center', fontsize=7, color=BLUE)
    ax.text(5.0, 0.85, 'layers, adapter', ha='center', fontsize=7, color=BLUE)
    ax.text(9.4, 0.85, 'audit', ha='center', fontsize=7, color=ORANGE)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_pipeline.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_pipeline.png'))
    plt.close(fig)
    print('  fig1_pipeline.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: BAC Layer Access Policy
# ═══════════════════════════════════════════════════════════════════════════
def fig2_bac_layers():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)

    layers = ['RAW', 'ENTITY', 'GRAPH', 'ABSTRACT']
    configs = [
        ('Normal\n$R \\leq 0.7$', [True, True, True, True], 'Default'),
        ('High Risk\n$0.7 < R \\leq 0.9$', [True, True, False, True], 'Challenger v1'),
        ('Escalation\n$R > 0.9$', [True, False, False, True], 'Challenger v2'),
    ]

    for ax_idx, (title, allowed, adapter) in enumerate(configs):
        ax = axes[ax_idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.8, 4.5)
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        for i, (layer, ok) in enumerate(zip(layers, allowed)):
            y = i * 1.05
            color = LIGHT_GREEN if ok else LIGHT_RED
            edge  = GREEN if ok else RED
            alpha_val = 1.0 if ok else 0.6

            rect = FancyBboxPatch((0.05, y), 0.9, 0.8,
                                   boxstyle='round,pad=0.05',
                                   facecolor=color, edgecolor=edge,
                                   linewidth=1.5, alpha=alpha_val)
            ax.add_patch(rect)

            label = layer if ok else f'{layer} [X]'
            txt_color = '#166534' if ok else '#991B1B'
            ax.text(0.5, y + 0.4, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=txt_color)

        # Adapter label
        ax.text(0.5, -0.5, f'Adapter: {adapter}', ha='center', fontsize=8,
                color=GRAY, fontstyle='italic')

    # Arrow under all three
    fig.text(0.2, 0.02, '', fontsize=1)
    arrow = fig.add_axes([0.15, 0.0, 0.7, 0.04])
    arrow.axis('off')
    arrow.annotate('', xy=(1, 0.5), xytext=(0, 0.5),
                   arrowprops=dict(arrowstyle='->', color=RED, lw=2.5))
    arrow.text(0.5, -0.8, 'Increasing sycophancy risk $R$  →',
               ha='center', fontsize=9, color=RED, transform=arrow.transAxes)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_bac_layers.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_bac_layers.png'))
    plt.close(fig)
    print('  fig2_bac_layers.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Sycophancy Rates Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def fig3_results_bar():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    conditions = ['Vanilla\nClaude', 'Static\nGuardrails*', 'Silicon\nMirror']
    rates = [20.0, 87.5, 0.0]
    colors = [RED, ORANGE, GREEN]
    edge_colors = ['#991B1B', '#9A3412', '#166534']

    bars = ax.bar(conditions, rates, color=colors, edgecolor=edge_colors,
                  linewidth=1.2, width=0.55, alpha=0.85, zorder=3)

    # Value labels on bars
    for bar, rate in zip(bars, rates):
        y = bar.get_height()
        label = f'{rate:.0f}%' if rate > 0 else '0%'
        ax.text(bar.get_x() + bar.get_width()/2, y + 2,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='#1F2937')

    ax.set_ylabel('Sycophancy Rate (%)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    ax.text(0.02, 0.98,
            '*Static Guardrails rate from\n simulated multi-turn scenarios',
            transform=ax.transAxes, fontsize=7, va='top', color=GRAY,
            fontstyle='italic')

    # n labels
    ax.text(0, -8, 'n=15\n(live)', ha='center', fontsize=7, color=GRAY)
    ax.text(1, -8, 'n=14\n(simulated)', ha='center', fontsize=7, color=GRAY)
    ax.text(2, -8, 'n=15\n(live)', ha='center', fontsize=7, color=GRAY)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_sycophancy_rates.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_sycophancy_rates.png'))
    plt.close(fig)
    print('  fig3_sycophancy_rates.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Risk Escalation Over Turns
# ═══════════════════════════════════════════════════════════════════════════
def fig4_risk_escalation():
    fig, ax = plt.subplots(figsize=(6, 3.8))

    turns = [1, 2, 3, 4, 5]
    risk_mirror = [0.19, 0.38, 0.58, 0.72, 0.85]
    risk_vanilla = [0.0, 0.0, 0.0, 0.0, 0.0]  # No tracking

    # Threshold zones
    ax.axhspan(0, 0.7, color=LIGHT_GREEN, alpha=0.35, zorder=0)
    ax.axhspan(0.7, 0.9, color=LIGHT_ORANGE, alpha=0.35, zorder=0)
    ax.axhspan(0.9, 1.0, color=LIGHT_RED, alpha=0.35, zorder=0)

    # Threshold lines
    ax.axhline(0.7, color=ORANGE, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(0.9, color=RED, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(5.15, 0.7, 'High risk\nthreshold', fontsize=7, color=ORANGE, va='center')
    ax.text(5.15, 0.9, 'Escalation\nthreshold', fontsize=7, color=RED, va='center')

    # Plot
    ax.plot(turns, risk_mirror, 'o-', color=BLUE, linewidth=2.2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, label='Silicon Mirror risk $R$',
            zorder=5)

    # Adapter annotations
    ax.annotate('Default adapter', xy=(2, 0.38), xytext=(0.5, 0.52),
                fontsize=7.5, color=GREEN, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.8))
    ax.annotate('Challenger v1', xy=(4, 0.72), xytext=(2.5, 0.82),
                fontsize=7.5, color=ORANGE, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))
    ax.annotate('Challenger v2', xy=(5, 0.85), xytext=(4.0, 0.95),
                fontsize=7.5, color=RED, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

    ax.set_xlabel('Conversation Turn', fontweight='bold')
    ax.set_ylabel('Sycophancy Risk Score $R$', fontweight='bold')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(turns)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_risk_escalation.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_risk_escalation.png'))
    plt.close(fig)
    print('  fig4_risk_escalation.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Sycophancy Pattern Taxonomy
# ═══════════════════════════════════════════════════════════════════════════
def fig5_pattern_taxonomy():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Validation-Before-Correction Pattern', ha='center',
            fontsize=12, fontweight='bold', color='#1F2937')

    # Vanilla pattern (left)
    ax.text(2.5, 5.1, 'Vanilla Claude', ha='center', fontsize=10,
            fontweight='bold', color=RED)

    steps_vanilla = [
        (2.5, 4.3, '1. Emotional validation\n"That\'s a great question!"', LIGHT_RED, RED),
        (2.5, 3.2, '2. Partial acknowledgment\n"In some sense, you\'re right..."', LIGHT_RED, RED),
        (2.5, 2.1, '3. Hedged correction\n"However, it\'s more nuanced..."', '#FEF3C7', ORANGE),
    ]

    for x, y, text, facecolor, edgecolor in steps_vanilla:
        rect = FancyBboxPatch((x - 2.0, y - 0.45), 4.0, 0.9,
                               boxstyle='round,pad=0.1',
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=1.3, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
                color='#1F2937')

    # Arrows between vanilla steps
    for y1, y2 in [(3.85, 3.65), (2.75, 2.55)]:
        ax.annotate('', xy=(2.5, y2), xytext=(2.5, y1),
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))

    # Result
    rect_v = FancyBboxPatch((0.5, 0.6), 4.0, 0.7,
                             boxstyle='round,pad=0.1',
                             facecolor='#FEE2E2', edgecolor=RED,
                             linewidth=1.8, linestyle='--')
    ax.add_patch(rect_v)
    ax.text(2.5, 0.95, 'User walks away feeling validated',
            ha='center', va='center', fontsize=8, fontweight='bold', color=RED)

    # Silicon Mirror pattern (right)
    ax.text(7.5, 5.1, 'Silicon Mirror', ha='center', fontsize=10,
            fontweight='bold', color=GREEN)

    steps_mirror = [
        (7.5, 4.3, '1. Direct correction\n"That\'s incorrect."', LIGHT_GREEN, GREEN),
        (7.5, 3.2, '2. Evidence presented\n"The data shows..."', LIGHT_GREEN, GREEN),
        (7.5, 2.1, '3. Constructive redirect\n"A more accurate view is..."', LIGHT_GREEN, GREEN),
    ]

    for x, y, text, facecolor, edgecolor in steps_mirror:
        rect = FancyBboxPatch((x - 2.0, y - 0.45), 4.0, 0.9,
                               boxstyle='round,pad=0.1',
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=1.3, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
                color='#1F2937')

    for y1, y2 in [(3.85, 3.65), (2.75, 2.55)]:
        ax.annotate('', xy=(7.5, y2), xytext=(7.5, y1),
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))

    rect_m = FancyBboxPatch((5.5, 0.6), 4.0, 0.7,
                             boxstyle='round,pad=0.1',
                             facecolor='#DCFCE7', edgecolor=GREEN,
                             linewidth=1.8, linestyle='--')
    ax.add_patch(rect_m)
    ax.text(7.5, 0.95, 'User receives accurate correction',
            ha='center', va='center', fontsize=8, fontweight='bold', color=GREEN)

    # VS divider
    ax.plot([5, 5], [1.0, 5.0], color=GRAY, linewidth=0.8, linestyle=':')
    ax.text(5, 0.3, 'vs', ha='center', fontsize=14, fontweight='bold', color=GRAY)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_pattern_taxonomy.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_pattern_taxonomy.png'))
    plt.close(fig)
    print('  fig5_pattern_taxonomy.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Risk Formula Component Breakdown
# ═══════════════════════════════════════════════════════════════════════════
def fig6_risk_components():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: Stacked bar showing risk components for a sample scenario
    ax = axes[0]
    turns = ['Turn 1', 'Turn 2', 'Turn 3']
    alpha_vals  = [0.15, 0.30, 0.38]  # agreeableness contribution
    sigma_vals  = [0.05, 0.10, 0.12]  # (1-skepticism) contribution
    gamma_vals  = [0.00, 0.12, 0.25]  # confidence_in_error contribution
    turn_bonus  = [0.00, 0.00, 0.00]  # B_turn (kicks in after turn 3)
    tactic_mult = [1.0,  1.3,  1.3]   # multiplier effect (shown as extra)

    # Base risk before multiplier
    base = [a + s + g for a, s, g in zip(alpha_vals, sigma_vals, gamma_vals)]
    # After multiplier
    after_mult = [b * m for b, m in zip(base, tactic_mult)]

    x = np.arange(len(turns))
    w = 0.45

    ax.bar(x, alpha_vals, w, label=r'$0.3\alpha$', color=BLUE, alpha=0.8)
    ax.bar(x, sigma_vals, w, bottom=alpha_vals, label=r'$0.2(1-\sigma)$', color=ORANGE, alpha=0.8)
    gamma_bottom = [a + s for a, s in zip(alpha_vals, sigma_vals)]
    ax.bar(x, gamma_vals, w, bottom=gamma_bottom, label=r'$0.3\gamma$', color=RED, alpha=0.8)

    # Multiplier effect (hatched)
    mult_extra = [am - b for am, b in zip(after_mult, base)]
    total_bottom = [a + s + g for a, s, g in zip(alpha_vals, sigma_vals, gamma_vals)]
    ax.bar(x, mult_extra, w, bottom=total_bottom, label=r'$M_\tau$ effect',
           color='#A855F7', alpha=0.5, hatch='//')

    ax.set_ylabel('Risk Score $R$')
    ax.set_title('Risk Component Breakdown', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(turns)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color=ORANGE, linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2.35, 0.72, '$R_{high}$', fontsize=7, color=ORANGE)
    ax.legend(fontsize=7, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: Tactic multiplier comparison
    ax2 = axes[1]
    tactics = ['None', 'Pleading', 'Moral\nEntreaty', 'Aggression', 'Framing',
               'Authority\nAppeal', 'Emotional\nManip.', 'Fake\nResearch']
    multipliers = [1.0, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5]
    colors_bar = [GRAY] + [BLUE]*2 + [ORANGE]*2 + [RED]*2 + ['#7C3AED']

    bars = ax2.barh(tactics, multipliers, color=colors_bar, alpha=0.8,
                     edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Tactic Multiplier $M_\\tau$')
    ax2.set_title('Persuasion Tactic Multipliers', fontsize=10, fontweight='bold')
    ax2.set_xlim(0.8, 1.65)
    ax2.axvline(1.0, color=GRAY, linestyle=':', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, val in zip(bars, multipliers):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}×', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_risk_components.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_risk_components.png'))
    plt.close(fig)
    print('  fig6_risk_components.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating figures...')
    fig1_pipeline()
    fig2_bac_layers()
    fig3_results_bar()
    fig4_risk_escalation()
    fig5_pattern_taxonomy()
    fig6_risk_components()
    print('Done. All figures saved to:', OUTPUT_DIR)

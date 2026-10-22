# Budgeted Discovery Agent: which Li-halide solid electrolytes are worth simulating?

Team: [add names] · OSI Hackathon 2026 · code: github.com/bruce249/osi-discovery-agent · video: [video link]

## Problem

Li3MX6 / Li2MX6 halide solid electrolytes; 236 candidates, 123 compositions; each costs a simulation, so the decision of what to simulate matters.

## Method

Strategy proposes 10 → CHGNet relaxation → same-model convex hull (MP references) → e_hull feedback → repeat until the budget is spent. Labels stay hidden from strategies.

Validation: MAE 14 meV/atom vs DFT over 399 reference compounds.

## Result

![Figure 1](../results/fig1_discovery.png)

With 90 simulations from a pool of 236 candidates (65 novel stable compositions hidden in it), GP-BO found the most novel stable compositions, 56.6 ± 2.8 versus 33.1 ± 2.8 for random: random search needs 1.7x as many simulations to match it (1.5x on all stable compositions), and the data-mined substitution baseline reaches 1.2x. The best LLM arm (LLM + GP hybrid) reached 1.3x on novel and 1.3x on all stable compositions. LLM agents were the fastest at rediscovering experimentally known compounds (LLM, basic prompt: 11.0 ± 0.8 of 13 vs 6.2 ± 2.0 for random), consistent with literature memory; the same prior pulls them toward known chemistry and away from novel compositions.

| strategy (@90 sims) | stable found | novel found | known rediscovered | acceleration vs random (novel) |
|---|---|---|---|---|
| LLM + GP hybrid | 58.0 ± 16.5 | 42.0 ± 18.5 | 11.0 ± 1.6 | 1.3 |
| LLM agent (reasoning model) | 54.3 ± 3.9 | 37.7 ± 4.7 | 11.0 ± 0.8 | 1.1 |
| LLM agent | 43.7 ± 1.7 | 29.0 ± 2.8 | 10.3 ± 1.7 | 0.9 |
| LLM, basic prompt | 41.0 ± 2.9 | 25.3 ± 4.0 | 11.0 ± 0.8 | 0.8 |
| GP-BO | 65.8 ± 4.6 | 56.6 ± 2.8 | 5.8 ± 1.5 | 1.7 |
| Data-mined substitution | 48.0 ± 0.0 | 39.0 ± 0.0 | 6.0 ± 0.0 | 1.2 |
| Ionic-radius heuristic | 49.0 ± 0.0 | 39.0 ± 0.0 | 6.0 ± 0.0 | 1.2 |
| Random | 43.2 ± 2.9 | 33.1 ± 2.8 | 6.2 ± 2.0 | 1.0 |

## New candidates (15/15 agreed by MACE-MP-0)

| formula | prototype | CHGNet e_hull | MACE e_hull | agreed | databases | literature |
|---|---|---|---|---|---|---|
| Li2CeBr6 | Li2ZrCl6-type | -0.133 | -0.032 | yes | absent (unchecked: Alexandria) | none found |
| Li3YbBr6 | Li3InCl6-type | -0.070 | -0.003 | yes | absent (unchecked: Alexandria) | none found |
| Li3Ho0.5Yb0.5Cl6 | Li3InCl6-type | -0.060 | -0.037 | yes | absent (unchecked: Alexandria, OQMD) | none found |
| Li3Yb0.5Dy0.5Cl6 | Li3YCl6-type | -0.054 | -0.017 | yes | absent (unchecked: Alexandria, OQMD) | none found |
| Li2ZrBr6 | Li2ZrCl6-type | -0.032 | +0.015 | yes | absent (unchecked: Alexandria, OQMD) | none found |

## Validation and limitations

ML potentials approximate DFT (CHGNet reference MAE 14 meV/atom; MACE 21 meV/atom); stability margins here are a few meV/atom, comparable to that error, so hits are predictions, not proof. Yb compounds come out most stable with CHGNet but much less so with MACE (Yb reference energies are known to be delicate in MP-trained models); Li2CeX6 needs Ce4+, which is not guaranteed to survive. Ordered approximants of partly disordered phases; 0 K thermodynamics only (no ionic conductivity or electrochemical window); campaigns replay precomputed evaluations. The brief's 0.03 eV/atom hit threshold makes 92% of the pool 'hits', so we used e_hull <= 0 (0.03/0.05 in summary.json). *'Novel' = not found in MP, GNoME, JARVIS, OQMD (partly reachable) and literature searches; the Alexandria server was unreachable during the run and is reported as unchecked.

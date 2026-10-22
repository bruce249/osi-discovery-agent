We built a budgeted discovery agent that decides which candidate Li-halide solid electrolytes are worth simulating, then opened up both AIs in its loop. A Gaussian process searching in the **internal feature space of the ML simulator itself** found the most stable materials of any strategy (69 of 84 in 90 simulations, vs 43 for random), and a causal test, a layer-by-layer probe and a two-domain experiment show *what* each AI actually knows: the simulator encodes stability before it simulates (and, in our teammates' companion probe, iron's redox state), while the LLM agent's edge is memorised literature.

**Live app** (try the LLM copilot, re-run the race): https://vignesh-82505--osi-discovery-agent-notebook.us-east.modal.run/

## Approach

- **Search space.** 236 candidates (123 compositions) built by substituting 22 trivalent and 5 tetravalent metals, Cl/Br, mixed-halide and mixed-metal orderings into Li3YCl6-, Li3InCl6- and Li2ZrCl6-type parents from Materials Project.
- **Simulator.** Each evaluation is a CHGNet relaxation scored by energy above a convex hull of 513 MP reference phases relaxed with the *same* potential (MAE 14 meV/atom vs DFT over 399 compounds). Stable = e_hull ≤ 0; novel = stable and absent from MP, GNoME, JARVIS, OQMD and a literature search. 886 relaxations fanned out over up to 80 Modal containers (3.4 min wall-clock).
- **The race.** Every strategy gets the same budget (90 simulations, rounds of 10, labels hidden): random, ICSD substitution statistics, ionic-radius heuristic, GP-BO on composition features, LLM agents (gpt-4.1-mini, gpt-5-mini) that justify each pick, and an LLM + GP hybrid. 3-10 seeds per arm.
- **Reading the simulator's mind** (probing protocol from our MACE-MP-0 interpretability study, [osi-draft](https://github.com/lntick/osi-draft)). One MACE-MP-0 forward pass on each *unrelaxed* candidate; ridge probes on Li-, metal- and halide-site features, tested on held-out chemistries, against a scrambled-weight copy of the network, composition features and shuffled labels. Then a GP searches in that latent space. Causal test: erase the 10 linear stability directions (iterative nullspace projection) and re-run the search.
- **Two domains.** Our reaction-optimisation benchmark ([osihackiitd](https://github.com/bruce249/osihackiitd), real Buchwald-Hartwig yield screens) ran the same question with an LLM agent that sees chemical names, and one that does not.
- **Checks.** MACE-MP-0 re-scoring of the top 15; phonons (CHGNet + phonopy, tiers from our [Lattice](https://github.com/dbajaj123/lattice-materials-discovery) studio).

## Results

![Two domains](https://raw.githubusercontent.com/bruce249/osi-discovery-agent/main/results/fig5_cross_domain.png)

| strategy (90 simulations) | stable found | novel found | known rediscovered |
|---|---|---|---|
| **GP on MACE latent** (5 seeds) | **69.2 ± 5.7** | 55.8 ± 6.0 | 9.0 |
| GP-BO, composition features | 65.8 ± 4.6 | **56.6 ± 2.8** | 5.8 |
| GP on scrambled-weight latent | 60.0 ± 6.4 | 48.6 ± 5.2 | 6.8 |
| LLM + GP hybrid (gpt-5-mini) | 58.0 ± 16.5 | 42.0 ± 18.5 | 11.0 |
| LLM agent (gpt-5-mini) | 54.3 ± 3.9 | 37.7 ± 4.7 | 11.0 |
| LLM agent (gpt-4.1-mini) | 43.7 ± 1.7 | 29.0 ± 2.8 | 10.3 |
| Random (10 seeds) | 43.2 ± 2.9 | 33.1 ± 2.8 | 6.2 |

- **Statistical surrogates find the new; LLMs find the known.** GP-BO needs 1.7× fewer simulations than random for novel stable compositions. LLM agents rediscover 10-11 of the 13 known compounds (random: 6) but are only 0.8-1.1× random on novel ones. On real reaction screens the same pattern holds: after 10 experiments the LLM reaches 0.84 of the best yield with chemical names, **0.74 with names hidden, the same as random (0.74)**. In both domains the LLM's advantage is prior knowledge, not search.
- **The simulator knows stability before it simulates.** Probes on unrelaxed-structure features predict post-relaxation stability on unseen chemistries: rank correlation 0.85, AUC 0.92, R² 0.67 (scrambled weights 0.80 / 0.90 / 0.41; composition features 0.72 / 0.84; shuffled labels AUC 0.49). Searching in this space found the most stable compositions of any arm and reached half of the novel ones fastest (1.85× random).

![Interpretability](https://raw.githubusercontent.com/bruce249/osi-discovery-agent/main/results/fig4_interp.png)

- **Where, whether it matters, what it means.** The signal peaks after MACE's first message-passing layer (0.86 vs 0.76 after the second); training adds it mainly on Li and halide sites (0.78-0.80 trained vs 0.54-0.62 scrambled), while metal identity alone already carries signal. Erasing 10 stability directions drops the probe AUC from 0.92 to 0.40 (10 random directions: 0.92) and the latent search from 69.2 to 61.2 stable (random erasure: 64.4; within seed spread, so consistent rather than conclusive). Textbook descriptors (metal radius and electronegativity, Br fraction, volume) explain only 42% of the stability direction.
- **Same network, second hidden skill.** Our teammates' probe of the same MACE-MP-0 ([osi-draft](https://github.com/lntick/osi-draft)) finds it learned iron's redox state from energies alone: high-spin Fe magnetic moments on unseen chemical systems at R² 0.75, vs 0.19 for geometry and 0.06 for scrambled weights; charge balance gives 0.49 and MACE adds 0.23 beyond geometry + charge balance. There too, ablating the probe direction moves energies no more than random directions do. Across both studies the lesson is the same: a probe shows the knowledge is present, a causal test shows whether it is used, and here it was usable enough to speed up discovery.
- **New candidates.** All 15 top novel hits stay within 0.05 eV/atom of the hull with MACE-MP-0, led by Li3Ho0.5Yb0.5Cl6, Li3Yb0.5Dy0.5Cl6, Li3Dy0.5Lu0.5Cl6 and Li3LuCl4Br2. Phonons: 2/15 free of imaginary modes, 12 with soft framework modes (-0.2 to -0.75 THz), 1 unstable (Li2ZrBr6).

**Limitations.** ML potentials approximate DFT and the stability margins (a few meV/atom) are comparable to their error; Yb energies are delicate in MP-trained models; ordered approximants of disordered phases; ionic conductivity not computed; the Alexandria database was unreachable, so "novel" means not found where we could search.

## Next steps

- DFT (r2SCAN) confirmation and larger-supercell phonons for the top candidates, then ionic-conductivity screening (NEB / MD) and synthesis of the best.
- Use the simulator's latent space as a free pre-screen: rank millions of hypothetical structures with one forward pass before relaxing any.
- Run the same probes on deeper universal potentials (MACE-MPA, Orb, UMA) to see whether stability is still encoded early, and steer along the stability direction to test causality inside the network.
- Ship the copilot to experimental groups as a planning tool: LLM for rationale and known chemistry, surrogate for the unknown.

Code, data and every number: https://github.com/bruce249/osi-discovery-agent

Every candidate battery material costs a simulation, so we built an agent that decides which ones are worth running, and then looked inside the AIs it relies on. The surprise was the simulator. Before it relaxes a single crystal, its internal features already tell you which candidates will turn out stable, and when we let the agent search in that space it found more stable materials than any other strategy we tried.

Live app (results, playground and a live LLM copilot): https://vignesh-82505--osi-discovery-agent-notebook.us-east.modal.run/

## Approach

We worked on lithium halide solid electrolytes, the Li3YCl6 / Li3InCl6 / Li2ZrCl6 family. Starting from three parent structures on Materials Project, we swapped in 27 metals, Cl or Br, and mixed metals or halides, giving 236 candidates (123 compositions). Each "experiment" is a CHGNet relaxation scored by its energy above a convex hull built from 513 reference phases relaxed with the same model. Against DFT, that setup is off by 14 meV/atom on average across 399 known compounds. A candidate counts as stable if it sits on or below the hull, and as novel if it is stable and we could not find it in Materials Project, GNoME, JARVIS, OQMD or the literature.

Every strategy got the same budget of 90 simulations, in rounds of 10, and never saw which materials were known or novel. We compared random picks, two chemistry heuristics, Bayesian optimisation (a Gaussian process on composition features), LLM agents (gpt-4.1-mini and gpt-5-mini) that explain each pick, and an LLM + GP hybrid. All relaxations ran on Modal, fanned out over up to 80 containers.

Then we opened up the simulator. We ran MACE-MP-0 once on each unrelaxed candidate and trained simple linear probes on its internal features to predict the stability CHGNet would later find. To make sure the probe was not fooling us, we tested only on chemistries it had never seen and compared against the same network with its weights scrambled, against plain composition features, and against shuffled labels. This follows the protocol from our teammates' MACE study ([osi-draft](https://github.com/lntick/osi-draft)). We then gave the Bayesian optimiser those features instead of composition features, and finally erased the stability directions from the network's features to see whether the search actually depended on them.

## Results

![Interpretability](https://raw.githubusercontent.com/bruce249/osi-discovery-agent/main/results/fig4_interp.png)

The probe works well on chemistries it never trained on: rank correlation 0.85 and AUC 0.92 for telling stable from unstable. Plain composition features reach 0.84, and shuffled labels sit at chance. The scrambled network is the interesting control. It ranks almost as well (0.90), because knowing which elements sit where already says a lot, but it gets the actual energies much worse (R² 0.41 against 0.67). Looking layer by layer, the signal is strongest right after the first message-passing layer, and training adds it mostly on the lithium and halide sites rather than on the metal. Textbook descriptors such as ionic radius and electronegativity explain only 42% of the stability direction.

Searching in that feature space paid off:

| strategy | stable found (of 84) | novel found (of 65) |
|---|---|---|
| GP on MACE's internal features | **69.2** | 55.8 |
| GP on composition features | 65.8 | **56.6** |
| GP on the scrambled network's features | 60.0 | 48.6 |
| LLM + GP hybrid (gpt-5-mini) | 58.0 | 42.0 |
| LLM agent (gpt-5-mini) | 54.3 | 37.7 |
| Random | 43.2 | 33.1 |

When we erased ten stability directions from the features, the probe dropped from AUC 0.92 to 0.40 and the search fell from 69 to 61 stable materials, about where the scrambled network sits. Erasing ten random directions left the probe untouched and cost the search less (64). The gap between those two is within the spread between seeds, so we read it as supporting evidence rather than proof. Our teammates found the same pattern for a different property: the same network, trained only on energies, predicts the magnetic moment of iron sites (and so their redox state) on unseen chemical systems with R² 0.75, against 0.19 for geometry alone and 0.06 for scrambled weights.

![Two domains](https://raw.githubusercontent.com/bruce249/osi-discovery-agent/main/results/fig5_cross_domain.png)

The LLM tells a different story. Its agents rediscovered 10 or 11 of the 13 known compounds in the pool, against 6 for random, but were no better than 0.8 to 1.1 times random at finding new ones. We saw the same thing on real lab data in our reaction-optimisation benchmark ([osihackiitd](https://github.com/bruce249/osihackiitd), Buchwald-Hartwig yield screens). After 10 experiments the LLM reached 84% of the best possible yield when it could see the chemical names, and 74% when the names were hidden, which is exactly what random search gets. Its head start is memory.

The best new candidates (Li3Ho0.5Yb0.5Cl6, Li3Yb0.5Dy0.5Cl6, Li3Dy0.5Lu0.5Cl6 and Li3LuCl4Br2) hold up with a second model: all 15 top picks stay within 0.05 eV/atom of the hull with MACE-MP-0. Phonon calculations (following our [Lattice](https://github.com/dbajaj123/lattice-materials-discovery) studio's checks) find 2 of the 15 completely free of soft modes, 12 with mild framework soft modes that need larger cells to settle, and one, Li2ZrBr6, that is clearly unstable.

These are predictions, not discoveries. The stability margins are a few meV/atom, comparable to the error of ML potentials, ordered cells stand in for partly disordered materials, and we did not compute ionic conductivity.

## Next steps

The obvious next step is DFT and larger-cell phonons for the top candidates, then conductivity screening and synthesis of the best one or two. On the interpretability side, a single forward pass costs about 0.4 s against roughly 6 s for a relaxation, so the simulator's own features could pre-screen far larger spaces before anything is relaxed. We also want to repeat the probes on deeper potentials (MACE-MPA, Orb, UMA) and test inside the network whether steering along the stability direction changes the energies it predicts.

Code and every number: https://github.com/bruce249/osi-discovery-agent

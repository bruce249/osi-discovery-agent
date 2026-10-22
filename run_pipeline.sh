#!/usr/bin/env bash
# Full pipeline on Modal. Every step caches to the Modal volume "osi-discovery-data" and is safe to re-run.
set -euo pipefail
export PYTHONIOENCODING=utf-8
M="modal run"

$M modal_app.py::smoke                                        # API checks + CHGNet/MACE timing
$M modal_app.py::step --name prototypes,candidates,mp_fetch   # MP parents, candidate pool, reference phases
$M modal_app.py::relax_batch --model chgnet                   # ~890 CHGNet relaxations, fanned out
$M modal_app.py::step --name hull                             # MLIP hulls, validation (Figure 2)
$M modal_app.py::step --name novelty,labels,features          # database + literature checks, hidden labels
$M modal_app.py::step --name campaign                         # all strategies incl. LLM agent
$M modal_app.py::step --name second_model --args prepare      # MACE task list for the top hits
$M modal_app.py::relax_batch --model mace --tasks-file data/cache/mace_tasks.json
$M modal_app.py::step --name second_model --args analyze
$M modal_app.py::step --name report                           # Figure 1, top_candidates.csv
$M modal_app.py::notebook_export                              # runs notebook.py top to bottom -> results/notebook.html

# pull everything the notebook needs back to this folder
modal volume get --force osi-discovery-data results .
modal volume get --force osi-discovery-data data .

# one-page write-up (standard library only), then print writeup/writeup.html to PDF from any browser
python writeup/build_writeup.py

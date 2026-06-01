# On the Exploitability of Fair Multi-Agent Reinforcement Learning

Code and paper for studying how **welfare-optimizing fair MARL** (SOTO/FEN-style,
optimizing a Generalized Gini Welfare over agents' returns) behaves when a single
member is **self-interested**.

**Summary of findings**
- **Diagnostic.** A single best-response defector exploits a fair team. In
  yield-based allocation games the over-share is `ρ = N` by arithmetic (analytical
  sanity checks; a faithful SOTO learner reproduces the same `N×`). The substantive
  evidence is in settings with *learned* contention: a collision-prone `claim` game
  (3.0×) and a Matthew-effect **gridworld** (1.9×, with seed variance).
- **Negative result.** A policy-level defense (reciprocity + adversarial
  co-training) fails; we prove (for the all-or-nothing model) that cooperators
  cannot deny a free-rider without wasteful levelling-down.
- **Positive result (with caveat).** **EKREM** instantiates the classic egalitarian
  (worst-off-first) allocation rule inside the MARL loop, restoring
  incentive-compatibility (free-ride ≈ 1 for N ≥ 4). It works by *removing agents'
  control over allocation* — it sidesteps, not solves, decentralized robustness.

## Layout
```
fairmarl/        # all experiment code (JAX + REINFORCE)
  fair_train.py     envs (claim / donate / matthew / fairmech) + 2-phase exploit test
  fair_soto.py      faithful SOTO (Self-/Team-Oriented sub-nets + annealing)
  fair_robust.py    failed policy-level defense (RECFAIR + adversarial)
  fair_solidify.py  scaling sweep over team size N
  fair_grid.py      FEN-style Matthew-effect gridworld
  fair_alloc.py     scripted exploit demo
  fair_bars.py / fair_arch.py   paper figures
paper/           # IEEE (UBMK) LaTeX source + compiled PDF + figures
```

## Reproduce
```bash
pip install -r requirements.txt          # jax, optax, flax, numpy, matplotlib, scipy
python -m fairmarl.fair_train  --env claim donate matthew fairmech   # exploit + fix (abstract)
python -m fairmarl.fair_soto   --env donate fairmech                 # faithful SOTO
python -m fairmarl.fair_solidify --ns 2 4 8 16                       # scaling
python -m fairmarl.fair_grid   --alloc matthew fairneed              # FEN gridworld
```
Each run trains a GGF-welfare fair team, then a best-response defector, and reports
the free-ride factor and Jain index.

## Paper
`paper/main.pdf` (IEEE conference / UBMK format). Build with
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## License
MIT (see `LICENSE`).

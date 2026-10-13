## Implementation

A description of the components in the OpenPlaceholder Cofolding -> FEP pipeline.

Each ligand (`A`, `B`, …, `n`) is run through its own `generator`. Every ligand
fans out into `n seeds × n diffusion samples` inference replicates that are each
gated by `validators.py`. The surviving replicates merge per ligand, and the
per-ligand results merge together into `selector`. From there the pipeline is
linear: `selector` → `transformations.py` → `mappers.py` → `simulators.py` →
`analyzers.py`.

```text
                              validators.py
                         ┌──> rep ──╫──┐
Ligand A ──> generator ──┼──> rep ──╫──┼──> merge A ─┐
                         └──> ... ──╫──┘             │
                         ┌──> rep ──╫──┐             │
Ligand B ──> generator ──┼──> rep ──╫──┼──> merge B ─┼──> selector  ──> transformations.py
                         └──> ... ──╫──┘             │                             │
                         ┌──> rep ──╫──┐             │                             │
Ligand n ──> generator ──┼──> rep ──╫──┼──> merge n ─┘                             v
                         └──> ... ──╫──┘                                      mappers.py
                                                                                   │
                                                                           AlchemicalNetwork
                                                                                   │
                         ┌─────────────────────────────────────────────────────────┘
                         │          simulators.py
                         ├──> edge A→B ──┐
                         ├──> edge B→C ──┼──> SimulationResults ──> analyzers.py ──> AnalysisResults
                         └──> ...     ───┘
```

`rep` = one inference replicate; each generator fans out into `n seeds × n diffusion samples`
replicates (the `...` stands for the remaining replicates). The `╫` column is the single
`validators.py` component: every replicate is gated individually as its line crosses the gate.

Component annotations:

- **generator** — generative suite of ligand–protein generative models. These can be
  cofolding models, docking models or anything else that will generate poses given some sort
  of protein context information.
- **validators.py** — collection of gates that will prevent any generated ligand–protein
  complexes from passing on in the pipeline. Any objects passing this step can be considered
  physically valid. These act on explicitly single-structure validations.
- **merge X** — the surviving replicates for ligand `X` are collected back together before
  cross-ligand selection.
- **selector** — contextualizes each pose hypothesis relative to the other pose
  hypotheses of the same ligand, and to the pose hypotheses of the full set of ligands. It
  attempts to pick a combination of pose hypotheses (n=1 per ligand) that optimizes the set of
  selection heuristics, e.g. making sure that ligand poses occupy the same 3D volume; core atoms
  overlap well; have the same ligand–protein interaction fingerprint.
- **transformations.py** — given the list of ligand–protein complexes, select and swap
  protein components and finetune structures as needed.
- **mappers.py** — generate an alchemical network given settings defined by the user;
  produce an `AlchemicalNetwork`.
- **simulators.py** — run each transformation (edge) of the `AlchemicalNetwork` and
  collect the executed `ProtocolDAG`s into a `SimulationResults`. Edges are independent,
  so this is a fan-out; the local simulator walks them sequentially on whichever hardware
  OpenMM selects.
- **analyzers.py** — pair each ligand's complex and solvent legs into a relative free
  energy, then estimate one absolute free energy per ligand across the network as a whole. Legs
  that failed, and ligands left unreachable by them, are dropped so that the rest of the
  network stays analyzable. Produces an `AnalysisResults` carrying the simulation results
  it read, the per-pair and per-ligand free energies, and the network statistics.

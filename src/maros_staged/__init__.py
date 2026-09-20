"""Stage-1 staged rebuild of Multi-Agent OS-SEI.

This package intentionally replaces the earlier all-at-once cross-domain model
(`maros_sei`) with the evidence-specialized, staged design described in
``docs/design/Multi-Agent_OS-SEI_Method_Design_and_Research_Status.md``.

Stage 1 implements the Shared Stem + Identity / Prototype / Reconstruction
agents and the complementarity gate. Stage 2 adds a disagreement-guided
Pseudo-Unknown Agent and an Open-Space Boundary Agent. Stage 3 adds explicit
directed communication and causal message interventions. Stage 4 adds a
constrained reliability router over isolated and communicated systems.
"""

__all__ = ["__version__"]
__version__ = "0.4.0"

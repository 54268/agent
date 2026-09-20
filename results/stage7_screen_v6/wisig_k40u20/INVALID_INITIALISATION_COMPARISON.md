# Invalid V6 attempt

The first V6 attempt was stopped after outer G0a.  Constructing the new
competence heads consumed the global PyTorch RNG stream and therefore changed
the subsequent Prototype Agent initialisation.  Its apparent improvement is
not a paired competence-head effect and must not be used for selection or
scientific claims.  The implementation now constructs auxiliary heads inside
an isolated RNG fork; the corrected run uses a fresh output directory.

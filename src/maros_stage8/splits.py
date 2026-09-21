"""Stage-8 entry point for the frozen, audited Stage-7 LCO primitives.

Re-exporting instead of copying keeps one tested implementation of sample
provenance and formal-Unknown exclusion while leaving all Stage-7 files
byte-for-byte unchanged.
"""
from maros_stage7.splits import (FORMAL_UNKNOWN_PURPOSE, ClassHoldoutFold,
                                 InnerEpisode, NestedLCOProtocol, OuterEpisode,
                                 ProvenanceSubset, assert_no_formal_unknown,
                                 assert_sample_disjoint, build_class_folds,
                                 build_inner_folds, build_nested_lco_protocol,
                                 protocol_manifest, validate_episode_isolation,
                                 validate_protocol_isolation)

__all__ = [
    "FORMAL_UNKNOWN_PURPOSE", "ClassHoldoutFold", "InnerEpisode",
    "NestedLCOProtocol", "OuterEpisode", "ProvenanceSubset",
    "assert_no_formal_unknown", "assert_sample_disjoint", "build_class_folds",
    "build_inner_folds", "build_nested_lco_protocol", "protocol_manifest",
    "validate_episode_isolation", "validate_protocol_isolation",
]


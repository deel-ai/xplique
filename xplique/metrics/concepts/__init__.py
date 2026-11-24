"""Standalone NumPy/SciPy concept-space evaluation metrics."""

from .geometry import (
    CoactiveSupportCoherence,
    concept_bank_babel_curve,
    concept_bank_babel_score,
    concept_bank_geometry_metrics,
)
from .matching import (
    ConceptAlignment,
    HungarianMatchingResult,
    MatchedCodeConsistency,
    compare_concept_banks,
)
from .prediction_fidelity import (
    BoxPredictions,
    ClassificationPredictionFidelity,
    DetectionPredictionFidelity,
    PosePredictionFidelity,
    PosePredictions,
    SemanticSegmentationPredictionFidelity,
)
from .reconstruction import ReconstructionError, ReconstructionR2
from .usage import ConceptSparsity, ConceptUsage, HoyerSparsity

__all__ = [
    "ReconstructionR2",
    "ReconstructionError",
    "ConceptUsage",
    "ConceptSparsity",
    "HoyerSparsity",
    "ClassificationPredictionFidelity",
    "DetectionPredictionFidelity",
    "PosePredictionFidelity",
    "SemanticSegmentationPredictionFidelity",
    "BoxPredictions",
    "PosePredictions",
    "ConceptAlignment",
    "MatchedCodeConsistency",
    "CoactiveSupportCoherence",
    "HungarianMatchingResult",
    "compare_concept_banks",
    "concept_bank_geometry_metrics",
    "concept_bank_babel_score",
    "concept_bank_babel_curve",
]

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .dino_clstoken_loss import DINOLoss
from .gram_loss import GramLoss
from .ibot_patch_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss, KoLeoLossDistributed
from .sigreg_loss import DistributedSIGReg
from .acquisition_orbit_deflation_loss import AcquisitionOrbitDeflationLoss
from .acquisition_tangent_projection import (
    acquisition_tangent_fraction,
    apply_acquisition_tangent_gradient_projection,
    build_acquisition_tangent_basis,
    project_onto_acquisition_tangent,
    rank_matched_random_tangent_basis,
)
from .nested_channel_innovation_loss import (
    ConditionalFeaturePredictor,
    NestedChannelInnovationLoss,
    NestedChannelInnovationWeights,
    conditional_innovation_residual,
    martingale_increment_orthogonality,
)
from .scout_kernel_delta_loss import (
    ScoutKernelDeltaTransportLoss,
    centered_cosine_kernel,
    cross_view_stable_kernel_delta,
)
from .conditional_morphology_graph_loss import (
    ConditionalEdgeGraphPredictor,
    ConditionalMorphologyGraphLoss,
    ConditionalMorphologyGraphWeights,
)

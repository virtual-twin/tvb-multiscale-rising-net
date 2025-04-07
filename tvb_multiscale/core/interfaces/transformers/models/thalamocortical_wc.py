# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits
from tvb.basic.neotraits._attr import NArray

from tvb_multiscale.core.interfaces.transformers.models.base import SpikesToRates, LinearRate
from tvb_multiscale.core.interfaces.transformers.models.integration import \
    ElephantSpikesHistogramRateLinearIntegration, ElephantSpikesRateLinearIntegration, \
    ElephantSpikesHistogramLinearIntegration,  ElephantSpikesHistogramRateLinearIntegration


class ThalamocorticalWCLinearRate(LinearRate):

    def _compute(self, input_buffer):
        # First make sure to sum all coupling variables,
        # in particular from:
        # 0. Isocortex
        # 1. Subcortex
        # (2. Specific Thalami, but this is one is not used)
        # input_buffer shape is assumed to be (proxy, time, vois)
        return super(ThalamocorticalWCLinearRate, self)._compute(input_buffer.sum(axis=-1))


class DefaultTVBtoSpikeNetTransformersThalamoCorticalWC(Enum):
    RATE = ThalamocorticalWCLinearRate
    # TODO: Need to potentially adjust all other transformers as well for summing up rates from all regions first!:
    # SPIKES = RatesToSpikesElephantPoisson
    # SPIKES_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    # SPIKES_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction
    # CURRENT = LinearCurrent


class DefaultSpikeNetToTVBTransformersThalamoCorticalWC(Enum):
    SPIKES = ElephantSpikesHistogramRateLinearIntegration
    SPIKES_TO_RATE = ElephantSpikesRateLinearIntegration
    SPIKES_TO_HIST = ElephantSpikesHistogramLinearIntegration
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRateLinearIntegration

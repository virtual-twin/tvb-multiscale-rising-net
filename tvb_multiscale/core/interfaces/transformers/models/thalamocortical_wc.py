# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits
from tvb.basic.neotraits._attr import NArray, Float

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


class ThalamocorticalWCSpikesToRate(HasTraits):

    baseline = Float(label="Baseline",
                     doc="Baseline of output",
                     required=True,
                     default=-0.5)

    transient = Float(label="Transient",
                      doc="Time length of transient increase of output",
                      required=True,
                      default=250.0)

    def configure(self):
        self._transient = int(self.transient/self.dt)

    def _compute(self, output_buffer, input_time):
        if input_time < self._transient:
            output_buffer = self.baseline + (output_buffer + self.baseline)/(self._transient - input_time)
        return output_buffer


class ThalamocorticalWCElephantSpikesHistogramRateLinearIntegration(
    ElephantSpikesHistogramRateLinearIntegration, ThalamocorticalWCSpikesToRate):

    def configure(self):
        ElephantSpikesHistogramRateLinearIntegration.configure(self)
        ThalamocorticalWCSpikesToRate.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        self.output_buffer = ElephantSpikesHistogramRateLinearIntegration._compute(self, input_buffer, *args, **kwargs)
        self.output_buffer = ThalamocorticalWCSpikesToRate._compute(self, self.output_buffer, self.input_time[0])
        return self.output_buffer


class ThalamocorticalWCElephantSpikesRateLinearIntegration(
    ElephantSpikesRateLinearIntegration, ThalamocorticalWCSpikesToRate):

    def configure(self):
        ElephantSpikesRateLinearIntegration.configure(self)
        ThalamocorticalWCSpikesToRate.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        self.output_buffer = ElephantSpikesRateLinearIntegration._compute(self, input_buffer, *args, **kwargs)
        self.output_buffer = ThalamocorticalWCSpikesToRate._compute(self, self.output_buffer, self.input_time[0])
        return self.output_buffer


class ThalamocorticalWCElephantSpikesHistogramLinearIntegration(
    ElephantSpikesHistogramLinearIntegration, ThalamocorticalWCSpikesToRate):

    def configure(self):
        ElephantSpikesHistogramLinearIntegration.configure(self)
        ThalamocorticalWCSpikesToRate.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        self.output_buffer = ElephantSpikesHistogramLinearIntegration._compute(self, input_buffer, *args, **kwargs)
        self.output_buffer = ThalamocorticalWCSpikesToRate._compute(self, self.output_buffer, self.input_time[0])
        return self.output_buffer


class DefaultSpikeNetToTVBTransformersThalamoCorticalWC(Enum):
    SPIKES = ThalamocorticalWCElephantSpikesHistogramRateLinearIntegration
    SPIKES_TO_RATE = ThalamocorticalWCElephantSpikesRateLinearIntegration
    SPIKES_TO_HIST = ThalamocorticalWCElephantSpikesHistogramLinearIntegration
    SPIKES_TO_HIST_RATE = ThalamocorticalWCElephantSpikesHistogramRateLinearIntegration

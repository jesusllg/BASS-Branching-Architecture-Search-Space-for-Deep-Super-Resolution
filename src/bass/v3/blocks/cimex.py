"""Consensus-Innovation Memory Exchange for interaction-aware BASS."""

from __future__ import annotations

import math
from collections.abc import Sequence

import tensorflow as tf

from ..config import (
    BRANCH_COUNT,
    CIMEX_MEMORY_RATIO,
    CIMEX_MIN_MEMORY_DIM,
    DEFAULT_CIMEX_GATE,
)

keras = tf.keras
layers = keras.layers


@keras.utils.register_keras_serializable(package="bass.v3")
class CIMEXLayer(layers.Layer):
    """Exchange compact innovation memories among symmetric BASS branches.

    The layer is branch-permutation equivariant because all projections are
    shared and every reduction treats the branch axis symmetrically. Spatial
    complexity is linear in the number of LR pixels for fixed prototype count.
    """

    def __init__(
        self,
        channels: int,
        prototypes: int,
        *,
        memory_dim: int | None = None,
        branch_count: int = BRANCH_COUNT,
        gate_init: float = DEFAULT_CIMEX_GATE,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.prototypes = int(prototypes)
        self.branch_count = int(branch_count)
        self.memory_dim = (
            max(CIMEX_MIN_MEMORY_DIM, round(channels * CIMEX_MEMORY_RATIO))
            if memory_dim is None
            else int(memory_dim)
        )
        self.gate_init = float(gate_init)
        self.epsilon = float(epsilon)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.prototypes <= 0:
            raise ValueError("prototypes must be positive")
        if self.branch_count < 2:
            raise ValueError("CIMEX requires at least two branches")
        if self.memory_dim <= 0:
            raise ValueError("memory_dim must be positive")
        if not 0.0 <= self.gate_init < 1.0:
            raise ValueError("gate_init must lie in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.innovation_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=self.epsilon,
            name="innovation_norm",
        )
        self.assignment_projection = layers.Dense(
            self.prototypes,
            name="assignment_projection",
        )
        self.value_projection = layers.Dense(
            self.memory_dim,
            name="value_projection",
        )
        self.query_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=self.epsilon,
            name="query_norm",
        )
        self.query_projection = layers.Dense(
            self.memory_dim,
            name="query_projection",
        )
        self.output_projection = layers.Dense(
            self.channels,
            use_bias=False,
            name="output_projection",
        )
        self.alpha = None

    def build(self, input_shape) -> None:
        if (
            not isinstance(input_shape, (list, tuple))
            or len(input_shape) != self.branch_count
        ):
            raise ValueError(
                f"CIMEX expects exactly {self.branch_count} branch input shapes"
            )
        reference = tf.TensorShape(input_shape[0])
        if reference.rank != 4:
            raise ValueError("CIMEX branch inputs must have rank 4 [B, H, W, C]")
        for shape in input_shape:
            current = tf.TensorShape(shape)
            if current.rank != 4:
                raise ValueError("CIMEX branch inputs must have rank 4 [B, H, W, C]")
            if current[-1] is not None and int(current[-1]) != self.channels:
                raise ValueError(
                    f"CIMEX expected {self.channels} channels, got {current[-1]}"
                )
            for axis in range(3):
                if (
                    reference[axis] is not None
                    and current[axis] is not None
                    and reference[axis] != current[axis]
                ):
                    raise ValueError("All CIMEX branch shapes must match")
        raw_gate = math.atanh(self.gate_init)
        stacked_shape = (
            reference[0],
            self.branch_count,
            reference[1],
            reference[2],
            self.channels,
        )
        flat_innovation_shape = (
            reference[0],
            self.branch_count,
            None,
            self.channels,
        )
        context_shape = (*stacked_shape[:-1], self.channels * 2)
        flat_context_shape = (
            reference[0],
            self.branch_count,
            None,
            self.channels * 2,
        )
        read_shape = (
            reference[0],
            self.branch_count,
            None,
            self.memory_dim,
        )
        self.innovation_norm.build(stacked_shape)
        self.assignment_projection.build(flat_innovation_shape)
        self.value_projection.build(flat_innovation_shape)
        self.query_norm.build(context_shape)
        self.query_projection.build(flat_context_shape)
        self.output_projection.build(read_shape)
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=keras.initializers.Constant(raw_gate),
            dtype="float32",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: Sequence[tf.Tensor]) -> tuple[tf.Tensor, ...]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != self.branch_count:
            raise ValueError(f"CIMEX expects exactly {self.branch_count} tensors")
        stacked = tf.stack(tuple(inputs), axis=1)
        consensus = tf.reduce_mean(stacked, axis=1, keepdims=True)
        innovations = stacked - consensus

        shape = tf.shape(stacked)
        batch = shape[0]
        height = shape[2]
        width = shape[3]
        pixels = height * width

        normalized_innovations = self.innovation_norm(innovations)
        flat_innovations = tf.reshape(
            normalized_innovations,
            (batch, self.branch_count, pixels, self.channels),
        )
        assignment_logits = self.assignment_projection(flat_innovations)
        assignments = tf.nn.softmax(assignment_logits, axis=2)
        values = self.value_projection(flat_innovations)
        memories = tf.einsum("bsnk,bsnd->bskd", assignments, values)

        memory_sum = tf.reduce_sum(memories, axis=1, keepdims=True)
        other_memories = (memory_sum - memories) / tf.cast(
            self.branch_count - 1, memories.dtype
        )

        consensus_per_branch = tf.repeat(consensus, repeats=self.branch_count, axis=1)
        query_context = tf.concat((stacked, consensus_per_branch), axis=-1)
        normalized_context = self.query_norm(query_context)
        flat_context = tf.reshape(
            normalized_context,
            (batch, self.branch_count, pixels, self.channels * 2),
        )
        queries = self.query_projection(flat_context)
        scores = tf.einsum("bsnd,bskd->bsnk", queries, other_memories)
        scores = scores / tf.cast(math.sqrt(self.memory_dim), scores.dtype)
        read_weights = tf.nn.softmax(scores, axis=-1)
        reads = tf.einsum("bsnk,bskd->bsnd", read_weights, other_memories)
        reads = tf.reshape(
            reads,
            (batch, self.branch_count, height, width, self.memory_dim),
        )
        projected_reads = self.output_projection(reads)
        centered_reads = projected_reads - tf.reduce_mean(
            projected_reads, axis=1, keepdims=True
        )

        gate = tf.cast(tf.math.tanh(self.alpha), stacked.dtype)
        updated = stacked + gate * tf.cast(centered_reads, stacked.dtype)
        return tuple(tf.unstack(updated, axis=1))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "prototypes": self.prototypes,
                "memory_dim": self.memory_dim,
                "branch_count": self.branch_count,
                "gate_init": self.gate_init,
                "epsilon": self.epsilon,
            }
        )
        return config


CIMEX = CIMEXLayer

"""
AlphaZero ResNet in Flax NNX: policy, value and margin heads, KataGo global pooling.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from engine.actions import ACTION_SIZE
from jaxgame.features import N_PLANES


# KataGo global pooling: whole-board summary.
def _pool(x: jnp.ndarray) -> jnp.ndarray:
    """(B, 9, 9, C) -> (B, 2C): board-wide mean and max of every channel."""
    return jnp.concatenate([x.mean(axis=(1, 2)), x.max(axis=(1, 2))], axis=-1)


class ResidualBlock(nnx.Module):
    """conv -> norm -> relu [-> + global bias] -> conv -> norm, add input, relu."""

    def __init__(self, channels: int, pooled: bool, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(channels, channels, (3, 3), padding="SAME", rngs=rngs)
        self.norm1 = nnx.BatchNorm(channels, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, (3, 3), padding="SAME", rngs=rngs)
        self.norm2 = nnx.BatchNorm(channels, rngs=rngs)
        self.global_bias = (  # KataGo global pooling, fed back as a bias
            nnx.Linear(2 * channels, channels, rngs=rngs) if pooled else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nnx.relu(self.norm1(self.conv1(x)))
        if self.global_bias is not None:
            y = y + self.global_bias(_pool(y))[:, None, None, :]
        y = self.norm2(self.conv2(y))
        return nnx.relu(y + x)


class QuoridorNet(nnx.Module):
    """Shared trunk feeding policy logits, a value in [-1, 1] and a margin estimate."""

    def __init__(self, *, rngs: nnx.Rngs, num_blocks: int = 8, channels: int = 96):
        self.stem_conv = nnx.Conv(N_PLANES, channels, (3, 3), padding="SAME", rngs=rngs)
        self.stem_norm = nnx.BatchNorm(channels, rngs=rngs)
        self.blocks = nnx.List(
            [
                ResidualBlock(channels, pooled=i % 3 == 2, rngs=rngs)  # every 3rd block
                for i in range(num_blocks)
            ]
        )
        # Policy: 1x1 conv to 2 planes, then dense to every action (walls are spatial).
        self.policy_conv = nnx.Conv(channels, 2, (1, 1), rngs=rngs)
        self.policy_norm = nnx.BatchNorm(2, rngs=rngs)
        self.policy_dense = nnx.Linear(2 * 9 * 9, ACTION_SIZE, rngs=rngs)
        # Value + margin (KataGo score target: loser's distance left), pooled summary
        self.value_hidden = nnx.Linear(2 * channels, channels, rngs=rngs)
        self.value_out = nnx.Linear(channels, 2, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """
        x: (B, 9, 9, N_PLANES) canonical planes.
        Returns (policy logits (B, 140), canonical frame; value (B,); margin (B,)).
        """
        x = nnx.relu(self.stem_norm(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)

        p = nnx.relu(self.policy_norm(self.policy_conv(x)))
        logits = self.policy_dense(p.reshape((p.shape[0], -1)))

        v = self.value_out(nnx.relu(self.value_hidden(_pool(x))))
        return logits, jnp.tanh(v[:, 0]), v[:, 1]


class TinyQuoridorNet(nnx.Module):
    """
    Lightweight MLP trunk suitable for fast CPU inference and pure Python embedding.
    Flattened canonical planes (9 * 9 * N_PLANES = 729) -> hidden layers -> policy, value, margin.
    """

    def __init__(self, *, rngs: nnx.Rngs, hidden_dim: int = 64, **kwargs):
        in_dim = 9 * 9 * N_PLANES
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.policy_out = nnx.Linear(hidden_dim, ACTION_SIZE, rngs=rngs)
        self.val_margin_out = nnx.Linear(hidden_dim, 2, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """
        x: (B, 9, 9, N_PLANES) canonical planes.
        Returns (policy logits (B, 140), value (B,), margin (B,)).
        """
        b = x.shape[0]
        h = x.reshape((b, -1))
        h = nnx.relu(self.fc1(h))
        h = nnx.relu(self.fc2(h))

        logits = self.policy_out(h)
        vm = self.val_margin_out(h)
        return logits, jnp.tanh(vm[:, 0]), vm[:, 1]


def create_model(
    seed: int = 0, num_blocks: int = 8, channels: int = 96, arch: str = "resnet"
) -> nnx.Module:
    """Build a freshly initialised network (ResNet or TinyQuoridorNet)."""
    rngs = nnx.Rngs(seed)
    if arch == "tiny":
        return TinyQuoridorNet(rngs=rngs, hidden_dim=channels)
    return QuoridorNet(rngs=rngs, num_blocks=num_blocks, channels=channels)


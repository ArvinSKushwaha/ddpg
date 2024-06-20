from typing import NamedTuple

import jax

from jax import numpy as np
from jaximal.core import Jaximal
from jaxtyping import Array, Bool, Float, PRNGKeyArray, UInt32


class MountainCarState(Jaximal):
    position: Float[Array, '1']  # [-1.2, 0.6]
    velocity: Float[Array, '1']  # [-0.07, 0.07]
    time: UInt32[Array, '1']  # [0, 999]


class MountainCarUpdate(NamedTuple):
    state: MountainCarState
    reward: Float[Array, '1']
    done: Bool[Array, '1']


class MountainCarAction(Jaximal):
    force: Float[Array, '1']  # [-1, 1]


class MountainCar:
    def init_state(self, key: PRNGKeyArray) -> MountainCarState:
        return MountainCarState(
            position=jax.random.uniform(key, shape=(1,), minval=-0.6, maxval=-0.4),
            velocity=np.array([0.0]),
            time=np.array([0], dtype=np.uint32),
        )

    def rand_action(
        self, state: MountainCarState, key: PRNGKeyArray
    ) -> MountainCarAction:
        return MountainCarAction(
            force=jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0),
        )

    def update(
        self, state: MountainCarState, action: MountainCarAction
    ) -> MountainCarUpdate:
        force = np.clip(action.force, -1.0, 1.0)
        velocity = state.velocity + force * 0.001 + np.cos(3 * state.position) * -0.0025
        position = state.position + state.velocity

        done = position >= 0.6
        position = np.clip(position, -1.2, 0.6)
        velocity = np.clip(velocity, -0.07, 0.07)
        new_state = MountainCarState(position, velocity, state.time + 1)

        return MountainCarUpdate(new_state, np.array([-1]), done)

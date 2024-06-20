from typing import Any, Callable, Protocol, cast

import jax
import optax

from jax import numpy as np
from jaximal.core import Jaximal, Static
from jaxtyping import Array, Bool, Float, Int32, PRNGKeyArray, Scalar
from tqdm.rich import tqdm

from mountain_car import MountainCar, MountainCarAction, MountainCarState
from utils import bind

# jax.config.update('jax_enable_x64', True)

N = 64

gamma = 0.99
tau = 1e-4
exploration_decay = 0.999
key = jax.random.key(102451756321645692)

key, state_key = jax.random.split(key)

system = MountainCar()

init_state = jax.vmap(system.init_state)
rand_action = jax.vmap(system.rand_action)
update = jax.vmap(system.update)

states = init_state(jax.random.split(state_key, N))

iterations = 30_000
progress_bar = tqdm(total=iterations)


class JaximalModule(Protocol):
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...


class Linear(Jaximal):
    weight: Float[Array, '{self.in_dim} {self.out_dim}']
    bias: Float[Array, '{self.out_dim}']

    in_dim: Static[int]
    out_dim: Static[int]

    @staticmethod
    def init_state(in_dim: int, out_dim: int, key: PRNGKeyArray) -> 'Linear':
        w_key, b_key = jax.random.split(key)

        weight = (
            jax.random.normal(w_key, shape=(in_dim, out_dim))
            / (in_dim * out_dim) ** 0.5
        )
        bias = jax.random.normal(b_key, shape=(out_dim,))

        return Linear(weight, bias, in_dim, out_dim)

    def forward(
        self, x: Float[Array, '{self.in_dim}']
    ) -> Float[Array, '{self.out_dim}']:
        return x @ self.weight + self.bias


class Activation(Jaximal):
    function: Static[Callable[[Array], Array]]

    def forward(self, x: Float[Array, 'n']) -> Float[Array, 'n']:
        return self.function(x)


class QSequential(Jaximal):
    modules: list[JaximalModule]

    def forward(
        self, state: MountainCarState, action: MountainCarAction
    ) -> Float[Array, '1']:
        x = np.concat([state.position, state.velocity, action.force])
        for module in self.modules:
            x = module.forward(x)

        return x


class ActionSequential(Jaximal):
    modules: list[JaximalModule]

    def forward(self, state: MountainCarState) -> MountainCarAction:
        x = np.concat([state.position, state.velocity])
        for module in self.modules:
            x = module.forward(x)

        return MountainCarAction(x)

    def forward_noisy(
        self,
        state: MountainCarState,
        key: PRNGKeyArray,
        noise: float,
    ) -> MountainCarAction:
        x = np.concat([state.position, state.velocity])
        for module in self.modules:
            x = module.forward(x)

        return MountainCarAction(x + jax.random.normal(key, shape=(1,)) * noise)


key, model_key = jax.random.split(key)


def make_model(key: PRNGKeyArray) -> tuple[QSequential, ActionSequential]:
    key, *q_model_keys = jax.random.split(key, 5)
    key, *action_model_keys = jax.random.split(key, 5)

    return QSequential(
        [
            Linear.init_state(3, 4, q_model_keys[0]),
            Activation(jax.nn.swish),
            Linear.init_state(4, 4, q_model_keys[1]),
            Activation(jax.nn.swish),
            Linear.init_state(4, 4, q_model_keys[2]),
            Activation(jax.nn.swish),
            Linear.init_state(4, 1, q_model_keys[3]),
            Activation(jax.nn.tanh),
        ]
    ), ActionSequential(
        [
            Linear.init_state(2, 4, action_model_keys[0]),
            Activation(jax.nn.swish),
            Linear.init_state(4, 4, action_model_keys[1]),
            Activation(jax.nn.swish),
            Linear.init_state(4, 4, action_model_keys[2]),
            Activation(jax.nn.swish),
            Linear.init_state(4, 1, action_model_keys[3]),
            Activation(jax.nn.tanh),
        ]
    )


q_model_state, action_model_state = make_model(model_key)
q_target_model_state, action_target_model_state = q_model_state, action_model_state


q_optimizer = optax.adam(1e-4)
action_optimizer = optax.adam(1e-3)

q_opt_state = q_optimizer.init(cast(optax.Params, q_model_state))
action_opt_state = action_optimizer.init(cast(optax.Params, action_model_state))


class OptimizationStep(Jaximal):
    q_model_state: QSequential
    action_model_state: ActionSequential

    q_target_model_state: QSequential
    action_target_model_state: ActionSequential

    states: MountainCarState

    q_opt_state: optax.OptState
    action_opt_state: optax.OptState

    exploration: float = 1


def q_model_loss(
    state: MountainCarState,
    new_state: MountainCarState,
    action: MountainCarAction,
    reward: Float[Array, 'n'],
    opt_step: OptimizationStep,
    q_model_state: QSequential,
) -> Float[Scalar, '']:
    current_step_q = q_model_state.forward(state, action)

    new_action = opt_step.action_target_model_state.forward(new_state)
    next_step_q = opt_step.q_target_model_state.forward(new_state, new_action)

    next_step_q = jax.lax.stop_gradient(next_step_q)

    bellman_loss = optax.l2_loss(reward + next_step_q * gamma, current_step_q).mean()
    return bellman_loss


def action_model_loss(
    states: MountainCarState,
    new_states: MountainCarState,
    actions: MountainCarAction,
    rewards: Float[Array, 'n'],
    opt_step: OptimizationStep,
    action_model_state: ActionSequential,
) -> Float[Scalar, '']:
    return -opt_step.q_target_model_state.forward(
        states, action_model_state.forward(states)
    )


def loop(
    i: int,
    loop_var: tuple[OptimizationStep, PRNGKeyArray],
) -> tuple[OptimizationStep, PRNGKeyArray]:
    opt_step, key = loop_var
    key, noise_key = jax.random.split(key)

    actions = jax.vmap(
        lambda x, key: opt_step.action_model_state.forward_noisy(
            x, key, opt_step.exploration
        ),
    )(opt_step.states, jax.random.split(noise_key, N))

    new_states, rewards, done = update(opt_step.states, actions)

    q_loss, q_grad = jax.value_and_grad(
        lambda model: jax.vmap(q_model_loss, in_axes=(0, 0, 0, 0, None, None))(
            opt_step.states,
            new_states,
            actions,
            rewards,
            opt_step,
            model,
        ).mean()
    )(opt_step.q_model_state)

    q_updates, new_q_opt_state = q_optimizer.update(
        q_grad, opt_step.q_opt_state, cast(optax.Params, opt_step.q_model_state)
    )

    new_q_model_state = optax.apply_updates(
        cast(optax.Params, opt_step.q_model_state), q_updates
    )

    action_loss, action_grad = jax.value_and_grad(
        lambda model: jax.vmap(action_model_loss, in_axes=(0, 0, 0, 0, None, None))(
            opt_step.states,
            new_states,
            actions,
            rewards,
            opt_step,
            model,
        ).mean()
    )(opt_step.action_model_state)

    action_updates, new_action_opt_state = action_optimizer.update(
        action_grad,
        opt_step.action_opt_state,
        cast(optax.Params, opt_step.action_model_state),
    )

    new_action_model_state = optax.apply_updates(
        cast(optax.Params, opt_step.action_model_state), action_updates
    )

    @bind(jax.debug.callback, ..., i, done.sum(), q_loss, action_loss)
    def _(
        i: int,
        x: Int32[Scalar, ''],
        q_loss: Float[Scalar, ''],
        action_loss: Float[Scalar, ''],
    ):
        progress_bar.update()

        if i % 50 == 0:
            print(f'{i:04d}: {x} {q_loss.item():.5f} {action_loss.item():.5f}')

    key, state_key = jax.random.split(key)
    rand_states = init_state(jax.random.split(state_key, N))

    new_states = MountainCarState(
        np.where(done, rand_states.position, new_states.position),
        np.where(done, rand_states.velocity, new_states.velocity),
        np.where(done, rand_states.time, new_states.time),
    )

    new_q_model_state = cast(QSequential, new_q_model_state)
    new_action_model_state = cast(ActionSequential, new_action_model_state)

    new_target_q_model_state = jax.tree.map(
        lambda a, b: a * tau + (1 - tau) * b,
        new_q_model_state,
        opt_step.q_target_model_state,
    )
    new_target_action_model_state = jax.tree.map(
        lambda a, b: a * tau + (1 - tau) * b,
        new_action_model_state,
        opt_step.action_target_model_state,
    )

    opt_step = OptimizationStep(
        new_q_model_state,
        new_action_model_state,
        new_target_q_model_state,
        new_target_action_model_state,
        new_states,
        new_q_opt_state,
        new_action_opt_state,
        exploration=opt_step.exploration * exploration_decay,
    )

    return opt_step, key


opt_step = OptimizationStep(
    q_model_state,
    action_model_state,
    q_target_model_state,
    action_target_model_state,
    states,
    q_opt_state,
    action_opt_state,
)


opt_step, key = jax.lax.fori_loop(
    0,
    iterations,
    loop,
    (opt_step, key),
)
progress_bar.close()

key, state_key = jax.random.split(key)

M = 100000
states = init_state(jax.random.split(state_key, M))
done = np.zeros((M, 1), dtype=np.bool)
rewards = np.zeros((M, 1))


def loop_validate(
    i: int, var: tuple[MountainCarState, Bool[Array, '{M} 1'], Float[Array, '{M} 1']]
) -> tuple[MountainCarState, Bool[Array, '{M} 1'], Float[Array, '{M} 1']]:
    states, done, rewards = var
    actions = jax.vmap(opt_step.action_target_model_state.forward)(states)
    updates = update(states, actions)

    states = updates.state
    done = done | updates.done
    rewards = rewards + gamma**i * updates.reward

    jax.debug.print('{}', np.count_nonzero(done))

    return states, done, rewards


jax.lax.fori_loop(0, 1000, loop_validate, (states , done, rewards))

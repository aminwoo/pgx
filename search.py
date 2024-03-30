from functools import partial
from time import time
import jax
import jax.numpy as jnp
import mctx

from pgx.bughouse import Bughouse, Action
from azresnet import AZResnet, AZResnetConfig


model = AZResnet(
    AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4,
        value_channels=8,
        num_policy_labels=64*77,
    )
)
x = jnp.ones((1, 8, 16, 32))
variables = model.init(jax.random.key(0), x, train=False)
forward = jax.jit(partial(model.apply, train=False))

seed = 42
batch_size = 1
key = jax.random.PRNGKey(seed)
key1, key2 = jax.random.split(key)
keys = jax.random.split(key2, batch_size)

# Load the environment
env = Bughouse()

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

def recurrent_fn(variables, rng_key: jnp.ndarray, action: jnp.ndarray, state):
    del rng_key
    current_player = state.current_player
    state = step_fn(state, action)

    policy_logits, value = forward(variables, state.observation)
    # mask invalid actions
    policy_logits = policy_logits - jnp.max(policy_logits, axis=-1, keepdims=True)
    policy_logits = jnp.where(state.legal_action_mask, policy_logits, jnp.finfo(policy_logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=policy_logits,
        value=value,
    )
    return recurrent_fn_output, state

def demo():
    state = init_fn(keys)
    policy_logits, value = forward(variables, state.observation)
    root = mctx.RootFnOutput(prior_logits=policy_logits, value=value, embedding=state)

    policy_output = mctx.gumbel_muzero_policy(
        params=variables,
        rng_key=key1,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=1000,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    return policy_output

jitted_demo = jax.jit(demo)
start = time()
out = jitted_demo()
print(time() - start)
#demo()
import time
import jax
import jax.numpy as jnp
import numpy as np 
import pgx 
from tqdm import tqdm

@jax.jit
def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)


def benchmark(iterations=100, seed=42):
    env = pgx.make('bughouse')
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)

    times = [] 
    for _ in tqdm(range(iterations)):
        key, subkey = jax.random.split(key)
        state = init_fn(subkey)
        while ~state.terminated:
            key, subkey = jax.random.split(key)
            action = act_randomly(subkey, state.observation, state.legal_action_mask)
            st = time.perf_counter()
            state = step_fn(state, action, key)
            end = time.perf_counter()
            if end - st < 1:
                times.append(end - st)
    return np.mean(times)

print(f'Mean step function execution time: {benchmark()}')
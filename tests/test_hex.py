import jax
import jax.numpy as jnp
from pgx.hex import Hex, init

env = Hex()
init = jax.jit(env.init)
step = jax.jit(env.step)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.curr_player == 1


def test_merge():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    state = step(state, 0)
    state = step(state, 11)
    state = step(state, 1)
    state = step(state, 12)
    state = step(state, 3)
    state = step(state, 13)
    state = step(state, 2)
    state = step(state, 22)
    # fmt: off
    expected = jnp.int16([
          3,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,
        -23, -23, -23,   0,   0,   0,   0,   0,   0,   0,   0,
        -23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])
    # fmt:on
    assert jnp.all(state.board == expected)


def test_terminated():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert not state.terminated
    for i in range(11):
        state = step(state, i * 11)
        state = step(state, i * 11 + 1)
    assert state.terminated

    state = init(key=key)
    for i in range(10):
        state = step(state, i)
        state = step(state, i + 11)
    state = step(state, 120)
    state = step(state, 21)
    assert state.terminated

    state = init(key=key)
    for i in range(10):
        state = step(state, i)
        state = step(state, i + 11)
    state = step(state, 10)
    assert not state.terminated
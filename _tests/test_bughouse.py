from _tests.board import BughouseBoard

import jax
import jax.numpy as jnp
import chess
from tqdm import tqdm
from pgx.bughouse import Bughouse, Action

def mirrorMoveUCI(uci_move):
    move = chess.Move.from_uci(uci_move)
    return mirrorMove(move).uci()


def mirrorMove(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion,
        move.drop,
    )

seed = 42
batch_size = 1
key = jax.random.PRNGKey(seed)

@jax.jit
def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)

env = Bughouse()
state = env.init(key)

init_fn = jax.jit(env.init)
step_fn = jax.jit(env.step)

simulations = 100000
for seed in tqdm(range(simulations)):
    key = jax.random.PRNGKey(seed)
    state = init_fn(key)

    board = BughouseBoard()
    while not (state.terminated | state.truncated).all():
        key, subkey = jax.random.split(key)

        action = act_randomly(subkey, state.observation, state.legal_action_mask)

        actions = [] 
        for i in range(2 * 64 * 78):
            if state.legal_action_mask[i]:
                actions.append(Action._from_label(i)._to_string()) 
        
        actions2 = [] 
        for i in board.boards[0].legal_moves:
            uci = i.uci()
            if board.boards[0].turn == False:
                actions2.append("0" + mirrorMoveUCI(uci))
            else:
                actions2.append("0" + uci)
        for i in board.boards[1].legal_moves:
            uci = i.uci()
            if board.boards[1].turn == False:
                actions2.append("1" + mirrorMoveUCI(uci))
            else:
                actions2.append("1" + uci)

        assert len(actions) == len(actions2)

        state = step_fn(state, action)  
        uci = Action._from_label(action)._to_string()
        if uci == "pass":
            continue
        if uci + "q" in actions2:
            uci += "q"

        board_num = int(uci[0])
        if board.boards[board_num].turn == False:
            board.push_uci(board_num, mirrorMoveUCI(uci[1:]))
        else:
            board.push_uci(board_num, uci[1:])


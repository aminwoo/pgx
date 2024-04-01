from _tests.board import BughouseBoard

import jax
import jax.numpy as jnp
import chess
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

simulations = 10000
for seed in range(simulations):
    key = jax.random.PRNGKey(seed)
    state = init_fn(key)

    board = BughouseBoard()
    while not (state.terminated | state.truncated).all():
        key, subkey = jax.random.split(key)

        actions = [] 
        for i in range(2 * 64 * 78):
            if state.legal_action_mask[i]:
                actions.append(Action._from_label(i)._to_string()) 
        
        actions2 = [] 
        for i in board.boards[0].legal_moves:
            uci = i.uci()
            if uci.endswith("q"):
                uci = uci[:-1]
            if board.boards[0].turn == False:
                actions2.append("0" + mirrorMoveUCI(uci))
            else:
                actions2.append("0" + uci)
        for i in board.boards[1].legal_moves:
            uci = i.uci()
            if uci.endswith("q"):
                uci = uci[:-1]
            if board.boards[1].turn == False:
                actions2.append("1" + mirrorMoveUCI(uci))
            else:
                actions2.append("1" + uci)

        print(state._promoted_pieces)
        print(state._board[0])
        #print(state._possible_piece_positions[1])
        print(state._to_fen())
        print(board.fen())
        print(state._pocket)
        print(sorted(actions))
        print(sorted(actions2))
        assert set(actions) == set(actions2)

        action = act_randomly(subkey, state.observation, state.legal_action_mask)
        print(Action._from_label(action)._to_string())
        print(Action._from_label(action))
        state = step_fn(state, action)  # state.reward (2,)
        uci = Action._from_label(action)._to_string()
        if uci == "pass":
            continue
        board_num = int(uci[0])
        if board.boards[board_num].turn == False:
            board.push_uci(board_num, mirrorMoveUCI(uci[1:]))
        else:
            board.push_uci(board_num, uci[1:])


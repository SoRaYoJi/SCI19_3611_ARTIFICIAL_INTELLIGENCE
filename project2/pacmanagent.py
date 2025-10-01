# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module import util

# Some environments expose Actions.directionToVector; we avoid that dependency by
# querying next position via state.generatePacmanSuccessor(action).


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        # Optional: remember last action for gentle tie-breaks
        self._last_action = None

    def _expected_distance(self, pac_pos, belief_state):
        """
        Compute sum over ghosts of the expected Manhattan distance
        from pac_pos to each ghost under its belief matrix.
        """
        if belief_state is None:
            return 0.0
        total = 0.0
        for b in belief_state:
            # If ghost eaten, b can be all zeros; skip to avoid NaNs
            s = b.sum()
            if s <= 0:
                continue
            # Compute expectation by iterating non-zero entries
            # (beliefs are typically sparse-ish after convergence)
            ex = 0.0
            # Iterate over indices where probability > 0 to save work
            xs, ys = (b > 0).nonzero()
            for x, y in zip(xs, ys):
                p = b[x, y]
                if p <= 0.0:
                    continue
                d = util.manhattanDistance((x, y), pac_pos)
                ex += p * d
            total += ex
        return float(total)

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        legal = state.getLegalPacmanActions()
        if not legal:
            return Directions.STOP

        # Prefer non-STOP actions if possible
        candidate_actions = [a for a in legal if a != Directions.STOP] or [Directions.STOP]

        # Evaluate expected distance if we take each action
        best = None
        best_score = None

        # Tie-break order (consistent, gentle bias)
        pref_order = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST, Directions.STOP]

        for a in candidate_actions:
            # Generate successor to know Pacman's next position under action a
            succ = state.generatePacmanSuccessor(a)
            if succ is None:
                # If the engine returns None (some frameworks do at terminal states), skip
                continue
            pac_next = succ.getPacmanPosition()

            score = self._expected_distance(pac_next, belief_state)

            # Small tie-break: prefer continuing the same direction, then by pref_order
            tie_bias = 0.0
            if a == self._last_action:
                tie_bias -= 1e-6
            tie_bias -= 1e-9 * pref_order.index(a) if a in pref_order else 0.0

            scored = (score, tie_bias)

            if best is None or scored < best_score:
                best = a
                best_score = scored

        self._last_action = best if best is not None else Directions.STOP
        return self._last_action

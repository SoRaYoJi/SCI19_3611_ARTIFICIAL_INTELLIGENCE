from pacman_module.game import Agent
from pacman_module.pacman import GameState
from pacman_module.util import manhattanDistance


def default_eval(state: GameState):
    """
    Evaluation function: ใช้คะแนนปัจจุบันของ state โดยตรง
    """
    return state.getScore()


class PacmanAgent(Agent):
    """
    Pacman agent ที่เล่นด้วย H-Minimax + Alpha-Beta pruning
    """

    def __init__(self, depth=2, eval_fn=default_eval):
        super().__init__()
        self.max_depth = int(depth)
        self.eval_fn = eval_fn

    def get_action(self, state: GameState):
        """
        คืน action ที่ดีที่สุดจากการทำ alpha-beta search
        """
        value, move = self._maximize(state, 0, 0, -float("inf"), float("inf"))
        return move

    # -----------------------------
    # alpha-beta recursive methods
    # -----------------------------

    def _maximize(self, state, depth, agent_id, alpha, beta):
        """
        Pacman (agent 0) เลือก action ที่ maximize ค่าประเมิน
        """
        if self._is_cutoff(state, depth):
            return self.eval_fn(state), None

        best_val, best_move = -float("inf"), None
        for act in state.getLegalActions(agent_id):
            nxt = state.generateSuccessor(agent_id, act)
            score, _ = self._minimize(nxt, depth, agent_id + 1, alpha, beta)

            if score > best_val:
                best_val, best_move = score, act

            # alpha-beta pruning
            alpha = max(alpha, best_val)
            if best_val >= beta:
                break

        return best_val, best_move

    def _minimize(self, state, depth, agent_id, alpha, beta):
        """
        Ghosts (agent 1..N) เลือก action ที่ minimize ค่าประเมิน
        """
        if self._is_cutoff(state, depth):
            return self.eval_fn(state), None

        worst_val, worst_move = float("inf"), None
        last_ghost = (agent_id == state.getNumAgents() - 1)

        for act in state.getLegalActions(agent_id):
            nxt = state.generateSuccessor(agent_id, act)

            if last_ghost:
                score, _ = self._maximize(nxt, depth + 1, 0, alpha, beta)
            else:
                score, _ = self._minimize(nxt, depth, agent_id + 1, alpha, beta)

            if score < worst_val:
                worst_val, worst_move = score, act

            # alpha-beta pruning
            beta = min(beta, worst_val)
            if worst_val <= alpha:
                break

        return worst_val, worst_move

    # -----------------------------
    # helper
    # -----------------------------

    def _is_cutoff(self, state, depth):
        """
        ตรวจว่าถึง state สิ้นสุดหรือยัง:
        - ชนะ, แพ้, หรือถึงความลึกสูงสุดแล้ว
        """
        return state.isWin() or state.isLose() or depth == self.max_depth
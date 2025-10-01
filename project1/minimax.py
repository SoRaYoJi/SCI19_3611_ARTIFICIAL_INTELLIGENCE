# minimax.py

from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState


def simple_eval(state: GameState) -> float:
    """ประเมินสถานะเกมด้วยคะแนนของ Pacman ตรง ๆ"""
    return state.getScore()


class PacmanAgent(Agent):
    """Pacman ที่ตัดสินใจด้วย Depth-limited Minimax"""

    def __init__(self, depth: str = "2"):
        super().__init__()
        self.max_depth = int(depth)

    # ---------- public API ----------
    def get_action(self, state: GameState):
        """
        เลือกแอ็กชันที่ให้ค่าสูงสุดจากราก (ตา Pacman)
        """
        best_move = Directions.STOP
        best_val = float("-inf")

        legal = state.getLegalActions(0)
        if not legal:
            return best_move

        for act in legal:
            nxt = state.generateSuccessor(0, act)
            val = self._minimax(nxt, ply=0, agent_id=1)  # ต่อด้วยตาผีตัวแรก
            if val > best_val:
                best_val, best_move = val, act

        return best_move

    # ---------- core search ----------
    def _minimax(self, state: GameState, ply: int, agent_id: int) -> float:
        """
        โหนดทั่วไป: สลับระหว่าง Pacman (max) และ Ghosts (min)
        - ply นับเป็นความลึกของ 'รอบ' (ครบทุกตัว 1 ครั้ง = เพิ่ม 1)
        - agent_id = 0 คือ Pacman, 1..N-1 คือ Ghosts
        """
        if self._cutoff(state, ply):
            return simple_eval(state)

        num_agents = state.getNumAgents()
        legal = state.getLegalActions(agent_id)
        if not legal:
            # ถ้าไม่มีแอ็กชัน ก็ประเมินสถานะปัจจุบัน
            return simple_eval(state)

        is_pacman = agent_id == 0

        if is_pacman:
            # ชั้น max
            value = float("-inf")
            for act in legal:
                nxt = state.generateSuccessor(agent_id, act)
                value = max(
                    value,
                    self._minimax(nxt, ply, (agent_id + 1) % num_agents),
                )
            return value
        else:
            # ชั้น min (ghost)
            value = float("inf")
            next_agent = (agent_id + 1) % num_agents
            # ถ้าถึงรอบของ Pacman อีกครั้ง ให้เพิ่ม ply (ลึกขึ้นอีก 1)
            next_ply = ply + 1 if next_agent == 0 else ply

            for act in legal:
                nxt = state.generateSuccessor(agent_id, act)
                value = min(value, self._minimax(nxt, next_ply, next_agent))
            return value

    # ---------- helpers ----------
    def _cutoff(self, state: GameState, ply: int) -> bool:
        """หยุดค้นเมื่อชนะ/แพ้ หรือถึงความลึกที่กำหนด"""
        return state.isWin() or state.isLose() or ply >= self.max_depth

# bfs.py
# Breadth-First Search agent for Pacman (PEP-8 compliant)

from collections import deque

from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    สร้างกุญแจ (hashable) แทนสถานะเกม:
    - ตำแหน่งแพ็กแมน
    - ตารางอาหาร (Grid)
    - รายการแคปซูล (tuple)
    """
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules()),
    )


class PacmanAgent(Agent):
    """
    Pacman agent ที่วางแผนเส้นทางด้วย Breadth-First Search (BFS)
    """

    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        """
        คืนทิศทาง 1 ก้าวตามลำดับแผน; ถ้าแผนว่างให้คำนวณด้วย BFS
        """
        if not self.moves:
            self.moves = self._bfs(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    # --------------------------
    # Core BFS
    # --------------------------
    def _bfs(self, start_state):
        """
        หาเส้นทางสั้นที่สุด (จำนวนก้าว) ไปยังสถานะชนะ (state.isWin()).
        คืนลิสต์ของ actions
        """
        if start_state.isWin():
            return []

        visited = {key(start_state)}
        q = deque([(start_state, [])])

        while q:
            cur, path = q.popleft()
            if cur.isWin():
                return path

            # generatePacmanSuccessors() -> [(next_state, action), ...]
            for nxt, act in cur.generatePacmanSuccessors():
                k = key(nxt)
                if k in visited:
                    continue
                visited.add(k)
                q.append((nxt, path + [act]))

        # ไม่พบทาง (ปกติไม่ควรเกิดในเลย์เอาต์มาตรฐาน)
        return []

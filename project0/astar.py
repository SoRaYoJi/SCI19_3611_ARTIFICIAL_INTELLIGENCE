# astar.py
# A* Search agent for Pacman (PEP-8 compliant)

import heapq
from itertools import count

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


def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def _food_positions(food_grid):
    """
    ดึงตำแหน่งอาหารทั้งหมดเป็นลิสต์ (x, y) จาก Grid
    รองรับทั้งที่มี .asList() หรือไม่มี
    """
    if hasattr(food_grid, "asList"):
        return list(food_grid.asList())

    pos = []
    width = len(food_grid)
    height = len(food_grid[0]) if width > 0 else 0
    for x in range(width):
        for y in range(height):
            if food_grid[x][y]:
                pos.append((x, y))
    return pos


def heuristic(state):
    """
    Heuristic แบบ admissible & simple:
    - เป้าหมายคือกินอาหาร/แคปซูลให้หมด → ใช้ max Manhattan
      จากตำแหน่งปัจจุบันไปยัง 'จุดเป้าหมายที่เหลือ' เป็น lower bound
    """
    pac = state.getPacmanPosition()
    foods = _food_positions(state.getFood())
    caps = list(state.getCapsules())
    targets = foods + caps
    if not targets:
        return 0
    return max(manhattan(pac, t) for t in targets)


class PacmanAgent(Agent):
    """
    Pacman agent ที่วางแผนด้วย A*:
    f(n) = g(n) + h(n), โดย g(n)=จำนวนก้าวสะสม, h(n)=heuristic ด้านบน
    """

    def __init__(self, args):
        self.moves = []
        self._tie = count()  # ตัวนับไว้เป็น tie-breaker ของคิว

    def get_action(self, state):
        if not self.moves:
            self.moves = self._astar(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    # --------------------------
    # Core A*
    # --------------------------
    def _astar(self, start_state):
        """
        หาเส้นทาง optimal (จำนวนก้าวต่ำสุด) ไปยังสถานะชนะ (state.isWin()).
        คืนลิสต์ของ actions
        """
        if start_state.isWin():
            return []

        start_k = key(start_state)
        g_best = {start_k: 0}

        # คิวเก็บทูเพิล: (f, tie, g, state, path)
        pq = []
        heapq.heappush(
            pq,
            (heuristic(start_state), next(self._tie), 0, start_state, []),
        )
        closed = set()

        while pq:
            f, _, g, cur, path = heapq.heappop(pq)
            cur_k = key(cur)
            if cur_k in closed:
                continue
            closed.add(cur_k)

            if cur.isWin():
                return path

            for nxt, act in cur.generatePacmanSuccessors():
                nxt_k = key(nxt)
                g2 = g + 1  # ค่าก้าวละ 1
                if nxt_k not in g_best or g2 < g_best[nxt_k]:
                    g_best[nxt_k] = g2
                    h2 = heuristic(nxt)
                    heapq.heappush(
                        pq,
                        (g2 + h2, next(self._tie), g2, nxt, path + [act]),
                    )

        # ไม่พบทาง
        return []

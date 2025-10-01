# Complete this class for all parts of the project

from pacman_module.game import Agent
import numpy as np
from pacman_module import util
from scipy.stats import binom


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.

            XXX: DO NOT MODIFY THE DEFINITION OF THESE VARIABLES
            # Doing so will result in a 0 grade.
        """

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        # Sensor noise parameters: Binomial(n, p) centered -> variance = n p (1-p).
        # We use p = 0.5 as in the evidence generator below.
        self.p = 0.5
        # Guard against division by zero and ensure integer n >= 1
        n_float = max(self.sensor_variance / (self.p * (1 - self.p)), 1.0)
        self.n = int(round(n_float))

        # Single free transition parameter (bias strength in [0,1])
        # If provided on the CLI as --ghostparam, use it; otherwise default.
        self.bias_strength = float(getattr(self.args, "ghostparam", 0.75))

        # Per-ghost-type scaling of the single parameter (keeps one free parameter overall)
        self._type_scale = {
            "scared": 1.0,       # strongest tendency to move away from Pacman
            "afraid": 0.6,       # moderate tendency
            "confused": 0.2      # close to random
        }

    # -------------------------
    # Utility helpers
    # -------------------------
    def _iter_positions(self):
        """Yield all (x,y) positions that are not walls."""
        width, height = self.walls.width, self.walls.height
        for x in range(width):
            for y in range(height):
                if not self.walls[x][y]:
                    yield (x, y)

    def _neighbors(self, pos):
        """Return non-wall 4-neighbors of pos within bounds (ghosts can move backward)."""
        x, y = pos
        cand = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        res = []
        width, height = self.walls.width, self.walls.height
        for nx, ny in cand:
            if 0 <= nx < width and 0 <= ny < height and not self.walls[nx][ny]:
                res.append((nx, ny))
        # In rare degenerate layouts with isolated cells, keep self-transition
        if not res:
            res = [pos]
        return res

    def _normalize(self, mat):
        s = mat.sum()
        if s > 0:
            mat /= s
        else:
            # Fallback to uniform over legal cells if everything is zero
            mask = np.logical_not(self._walls_numpy())
            mat[mask] = 1.0 / np.count_nonzero(mask)
        return mat

    def _walls_numpy(self):
        """Return a boolean numpy mask True for walls, shape [W,H]."""
        W, H = self.walls.width, self.walls.height
        arr = np.zeros((W, H), dtype=bool)
        for x in range(W):
            for y in range(H):
                if self.walls[x][y]:
                    arr[x, y] = True
        return arr

    # -------------------------
    # Models
    # -------------------------
    def _get_sensor_model(self, pacman_position, evidence):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The sensor model represented as a 2D numpy array of
        size [width, height].
        The element at position (w, h) is the probability
        P(E_t=evidence | X_t=(w, h))
        """
        W, H = self.walls.width, self.walls.height
        S = np.zeros((W, H), dtype=float)

        # For each possible ghost location x=(w,h), compute the likelihood of observing `evidence`:
        # Evidence model: E = d(X, Pacman) + (K - n*p), with K ~ Binomial(n, p), p=0.5
        # => P(E=e | X=x) = P(K = e - d(x) + n*p) if that integer lies in [0, n], else 0.
        for (w, h) in self._iter_positions():
            d = util.manhattanDistance((w, h), pacman_position)
            # evidence is typically an integer; keep robust to float rounding
            k_star = evidence - d + self.n * self.p
            k_rounded = int(round(k_star))
            if 0 <= k_rounded <= self.n and abs(k_star - k_rounded) < 1e-8:
                S[w, h] = binom.pmf(k_rounded, self.n, self.p)
            else:
                S[w, h] = 0.0

        # Zero out walls just in case
        S[self._walls_numpy()] = 0.0
        return S

    def _get_transition_model(self, pacman_position):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        The element at position (w1, h1, w2, h2) is the probability
        P(X_t+1=(w1, h1) | X_t=(w2, h2))
        """
        W, H = self.walls.width, self.walls.height
        T = np.zeros((W, H, W, H), dtype=float)

        # Effective bias strength for the selected ghost type (single free parameter overall)
        scale = self._type_scale.get(self.ghost_type, 0.6)
        beta = np.clip(self.bias_strength * scale, 0.0, 1.0)

        for (x, y) in self._iter_positions():
            nbrs = self._neighbors((x, y))

            # Compute how desirable each neighbor is: we favor moves that INCREASE distance to Pacman.
            d_curr = util.manhattanDistance((x, y), pacman_position)
            desirabilities = []
            for (nx, ny) in nbrs:
                d_next = util.manhattanDistance((nx, ny), pacman_position)
                desirabilities.append(d_next - d_curr)  # positive => moving away

            desirabilities = np.array(desirabilities, dtype=float)

            # Soft preference: assign probability mass beta to the neighbors that maximize "moving away"
            # and spread the remaining (1-beta) uniformly across all legal neighbors.
            if desirabilities.size == 0:
                # Isolated cell (shouldn't happen), self-loop
                T[x, y, x, y] = 1.0
                continue

            max_gain = desirabilities.max()
            prefer_mask = (desirabilities == max_gain)
            n_pref = prefer_mask.sum()
            n_all = len(nbrs)

            if n_pref == 0:
                # No improvement possible -> all uniform
                probs = np.ones(n_all, dtype=float) / n_all
            else:
                # Put beta mass on preferred neighbors, uniformly; rest uniformly on all
                probs = np.full(n_all, (1.0 - beta) / n_all, dtype=float)
                probs[prefer_mask] += beta / n_pref

            # Fill T for transitions from (x,y) to each neighbor
            for (prob, (nx, ny)) in zip(probs, nbrs):
                T[nx, ny, x, y] += prob

        return T

    def _get_updated_belief(self, belief, evidences, pacman_position,
            ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        and the previous belief states before receiving the evidences,
        returns the updated list of belief states about ghosts positions

        Arguments:
        ----------
        - `belief`: A list of Z belief states at state x_{t-1}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze.
               Matrices filled with zeros must be returned for eaten ghosts.
        """

        # Build models once per step
        T = self._get_transition_model(pacman_position)

        new_beliefs = []
        W, H = self.walls.width, self.walls.height
        walls_mask = self._walls_numpy()

        for z, (b_prev, evi, eaten) in enumerate(zip(belief, evidences, ghosts_eaten)):
            if eaten:
                new_beliefs.append(np.zeros_like(b_prev))
                continue

            # 1) Prediction step: b_pred(x') = sum_x T(x'|x) * b_prev(x)
            # Implement efficiently by distributing from each previous state.
            b_pred = np.zeros((W, H), dtype=float)
            for x in range(W):
                for y in range(H):
                    if walls_mask[x, y]:
                        continue
                    mass = b_prev[x, y]
                    if mass <= 0.0:
                        continue
                    # Distribute to neighbors according to T[:, :, x, y]
                    b_pred += mass * T[:, :, x, y]

            # 2) Update step with sensor model
            S = self._get_sensor_model(pacman_position, evi)
            b_post = b_pred * S

            # 3) Normalize and clean walls
            b_post[walls_mask] = 0.0
            b_post = self._normalize(b_post)

            new_beliefs.append(b_post)

        return new_beliefs

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        belief = self._get_updated_belief(self.beliefGhostStates, evidences,
                                          pacman_position, ghosts_eaten)
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n*self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        pass

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A belief state.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(evidence,
                                                   state.getPacmanPosition(),
                                                   state.data._eaten[1:])
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence

from pacman_module.game import Agent
from pacman_module.pacman import GameState
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.depth = 3

    def get_action(self, game_state: GameState):
        """
        Returns the minimax action using a heuristic evaluation function
        """
        memo = {}
        best_action, _ = self.minimax(game_state, 0, 0, memo)
        return best_action

    def minimax(self, game_state, depth, agent_index, memo):
        state_key = (game_state, depth, agent_index)
        if state_key in memo:
            return memo[state_key]
        if (game_state.isWin() or game_state.isLose() or
                depth == self.depth * game_state.getNumAgents()):
            return None, self.heuristic(game_state)

        num_agents = game_state.getNumAgents()
        next_agent_index = (agent_index + 1) % num_agents
        next_depth = depth + 1

        if agent_index == 0:  # Pacman's turn
            successors = game_state.generatePacmanSuccessors()
        else:  # Ghost's turn
            successors = game_state.generateGhostSuccessors(agent_index)

        if not successors:
            return None, self.heuristic(game_state)

        scores = []
        for successor_state, action in successors:
            _, score = self.minimax(
                successor_state, next_depth, next_agent_index, memo)
            scores.append(score)

        if agent_index == 0:  # Pacman (Maximizer)
            best_score = max(scores)
            best_indices = [
                index for index, score in enumerate(scores)
                if score == best_score
            ]
            chosen_index = best_indices[0]
            result = successors[chosen_index][1], best_score
            memo[state_key] = result
            return result
        else:  # Ghost (Minimizer)
            best_score = min(scores)
            best_indices = [
                index for index, score in enumerate(scores)
                if score == best_score
            ]
            chosen_index = best_indices[0]
            result = successors[chosen_index][1], best_score
            memo[state_key] = result
            return result

    def heuristic(self, game_state):
        """
        Computes a heuristic value for a given game state.
        """
        pacman_position = game_state.getPacmanPosition()
        ghost_positions = game_state.getGhostPositions()
        food_grid = game_state.getFood()
        food_list = food_grid.asList()

        # Start with the current score
        score = game_state.getScore()

        # Ghost distance
        if ghost_positions:
            min_ghost_dist = min([
                manhattanDistance(pacman_position, pos)
                for pos in ghost_positions
            ])
            if min_ghost_dist > 0:
                score -= 10.0 / min_ghost_dist

        # Food distance
        if food_list:
            min_food_dist = min([
                manhattanDistance(pacman_position, food)
                for food in food_list
            ])
            score -= min_food_dist

        return score

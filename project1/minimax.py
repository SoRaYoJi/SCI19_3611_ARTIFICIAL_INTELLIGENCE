from pacman_module.game import Agent
from pacman_module.pacman import GameState


class PacmanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.depth = 3

    def get_action(self, game_state: GameState):
        """
        Returns the minimax action from the current game_state.
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
            return None, game_state.getScore()

        num_agents = game_state.getNumAgents()
        next_agent_index = (agent_index + 1) % num_agents
        next_depth = depth + 1

        if agent_index == 0:  # Pacman's turn
            successors = game_state.generatePacmanSuccessors()
        else:  # Ghost's turn
            successors = game_state.generateGhostSuccessors(agent_index)

        if not successors:
            return None, game_state.getScore()

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
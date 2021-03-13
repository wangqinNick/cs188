# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


def getMinDistance(newPos, newFood):
    min_distance = 999999
    for i in range(len(newFood[0])):
        for j in range(len(newFood[1])):
            if util.manhattanDistance(newPos, (i, j)) < min_distance:
                min_distance = util.manhattanDistance(newPos, (i, j))
    return min_distance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()  # new pacman position
        newFood = childGameState.getFood()  # new remaining food
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        newGhostPosition = childGameState.getGhostPosition(agentIndex=1)
        distance_to_ghost = util.manhattanDistance(newPos, newGhostPosition)
        new_food_num = childGameState.getNumFood()
        distance_to_nearest_food = getMinDistance(newPos=newPos, newFood=newFood)
        num_pacman = childGameState.getNumAgents()
        if distance_to_ghost <= 2:
            distance_penalty = -100
        else:
            distance_penalty = 1
        evaluated_score = - 5 * new_food_num + 10 * num_pacman + distance_penalty
        return evaluated_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    "If requisite no. of searches complete, evaluation function"
    """
    def max_value(self, gameState, current_depth):
        current_depth += 1
        if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
            return self.evaluationFunction(gameState)
        legal_actions = gameState.getLegalActions(0)
        successors = [gameState.getNextState(0, action) for action in legal_actions]
        v = -float('inf')
        for successor in successors:
            v = max(v, self.min_value(gameState=successor,
                                      current_depth=current_depth,
                                      agent_index=1))
        return v

    def min_value(self, gameState, current_depth, agent_index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legal_actions = gameState.getLegalActions(agent_index)
        successors = [gameState.getNextState(agent_index, action) for action in legal_actions]
        v = float('inf')
        for successor in successors:
            if agent_index + 1 >= gameState.getNumAgents():
                v = min(v, self.max_value(gameState=successor,
                                          current_depth=current_depth + 1))
            else:
                v = min(v, self.min_value(gameState=successor,
                                          current_depth=current_depth,
                                          agent_index=agent_index + 1))
            return v
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        """    ghostIndex = [i for i in range(1, gameState.getNumAgents())]

        def terminal(state, depth):
            return state.isWin() or state.isLose() or self.depth == depth

        def max_value(state, depth):
            if terminal(state, depth):
                return self.evaluationFunction(state)
            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.getNextState(0, action), depth, 1))
            return v

        def min_value(state, depth, ghost):
            if terminal(state, depth):
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(ghost):
                if ghost == ghostIndex[-1]:
                    v = min(v, max_value(state.getNextState(ghost, action), depth + 1))
                else:
                    v = min(v, min_value(state.getNextState(ghost, action), depth, ghost + 1))
            return v

        result = [(action, min_value(gameState.getNextState(0, action), 0, 1)) for action in
                  gameState.getLegalActions()]
        result.sort(key=lambda x: x[1])
        return result[-1][0]"""
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        def isTerminate(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghostIndex):  # minimizer

            if isTerminate(state, d):
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == GhostIndex[-1]:
                    v = min(v, max_value(state.getNextState(ghostIndex, action), d + 1))
                else:
                    v = min(v, min_value(state.getNextState(ghostIndex, action), d, ghostIndex + 1))
            # print(v)
            return v

        def max_value(state, d):  # maximizer

            if isTerminate(state, d):
                return self.evaluationFunction(state)

            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.getNextState(0, action), d, 1))
            # print(v)
            return v

        res = [(action, min_value(gameState.getNextState(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def isTerminate(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def value(state, alpha, beta, depth, ghostIndex):
            if isTerminate(state, depth):
                return self.evaluationFunction(state)
            print(ghostIndex)
            if ghostIndex == state.getNumAgents():
                ghostIndex = 0
            if ghostIndex == 0:
                return max_value(state=state, alpha=alpha, beta=beta,
                                 depth=depth)
            else:
                return min_value(state=state, alpha=alpha, beta=beta,
                                 depth=depth, ghostIndex=ghostIndex)

        def max_value(state, alpha, beta, depth, ghostIndex=0):
            v = -float('inf')
            actions = state.getLegalActions(ghostIndex)
            childrenState = state.getNextState(ghostIndex, actions)
            ghostIndex += 1
            for childState in childrenState:
                v = max(v, value(childState, alpha, beta, depth, ghostIndex))
                if v >= beta: return v
                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth, ghostIndex):
            v = float('inf')
            actions = state.getLegalActions(ghostIndex)
            childrenState = state.getNextState(ghostIndex, actions)
            ghostIndex += 1
            for childState in childrenState:
                v = min(v, value(childState, alpha, beta, depth, ghostIndex))
                if v <= alpha: return v
                beta = min(beta, v)
            return v

        """def alphabeta(state):

            v = -float('inf')
            act = None
            A = -float('inf')
            B = float('inf')

            for action in state.getLegalActions(0):  # maximizing
                tmp = value(state=gameState.getNextState(0, action),
                            depth=0,
                            ghostIndex=1,
                            alpha=A, beta=B)

                if v < tmp:  # same as v = max(v, tmp)
                    v = tmp
                    act = action

                if v > B:  # pruning
                    return v
                A = max(A, tmp)

            return act

        return alphabeta(gameState)"""

        """def min_val(curr_state, curr_depth, agent_index, a, b):
            ghost_num = curr_state.getNumAgents() - 1
            if curr_depth == self.depth or curr_state.isWin() or curr_state.isLose():
                return self.evaluationFunction(curr_state)
            val = float("+inf")
            for action in curr_state.getLegalActions(agent_index):
                succ_state = curr_state.getNextState(agent_index, action)
                if agent_index == ghost_num:
                    val = min(val, max_val(succ_state, curr_depth + 1, a, b))
                    if val < a:
                        return val
                    b = min(b, val)
                else:
                    val = min(val, min_val(succ_state, curr_depth, agent_index + 1, a, b))
                    if val < a:
                        return val
                    b = min(b, val)

            return val

        def max_val(curr_state, curr_depth, a, b):
            if curr_depth == self.depth or curr_state.isWin() or curr_state.isLose():
                return self.evaluationFunction(curr_state)
            val = float("-inf")
            for action in curr_state.getLegalActions(0):
                val = max(min_val(curr_state.getNextState(0, action), curr_depth, 1, a, b), val)
                if val > b:
                    return val
                a = max(a, val)
            return val"""

        result = float("-inf")
        Action = "Still"
        a = float("-inf")
        b = float("+inf")
        for action in gameState.getLegalActions(0):
            tmp = value(gameState.getNextState(0, action), depth=0, ghostIndex=1,
                        alpha=a, beta=b)
            if tmp > result:
                Action = action
                result = tmp
            if result > b:
                return result
            a = max(a, result)
        return Action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

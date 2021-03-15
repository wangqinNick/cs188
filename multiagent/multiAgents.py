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
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        def isTerminate(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def max_value(state, alpha, beta, depth, ghostIndex=0):
            if isTerminate(state, depth): return self.evaluationFunction(state)

            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, min_value(state=state.getNextState(0, action),
                                     alpha=alpha,
                                     beta=beta,
                                     depth=depth,
                                     ghostIndex=1))
                if v > beta: return v
                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth, ghostIndex):
            if isTerminate(state, depth): return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == GhostIndex[-1]:
                    v = min(v, max_value(state=state.getNextState(ghostIndex, action),
                                         alpha=alpha,
                                         beta=beta,
                                         depth=depth + 1))
                else:
                    v = min(v, min_value(state=state.getNextState(ghostIndex, action),
                                         alpha=alpha,
                                         beta=beta,
                                         depth=depth,
                                         ghostIndex=ghostIndex + 1))
                if v < alpha: return v
                beta = min(beta, v)
            return v

        def alpha_beta():
            v = -float('inf')
            move = None
            A = -float('inf')
            B = float('inf')
            for action in gameState.getLegalActions(0):
                tmp = min_value(state=gameState.getNextState(0, action),
                                alpha=A,
                                beta=B,
                                depth=0,
                                ghostIndex=1)
                if v < tmp:  # updating best move
                    v = tmp
                    move = action
                if v > B:  # pruning
                    continue
                A = max(A, v)  # updating best value so far
            return move

        return alpha_beta()

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
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        def isTerminate(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def max_value(state, d, ghostIndex=0):
            if isTerminate(state, d): return self.evaluationFunction(state)
            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, exp_value(state.getNextState(0, action), d, 1))
            return v

        def exp_value(state, d, ghostIndex):
            if isTerminate(state, d): return self.evaluationFunction(state)
            v = 0
            p = 1.0 / len(state.getLegalActions(ghostIndex))
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == GhostIndex[-1]:
                    v += p * max_value(state.getNextState(ghostIndex, action), d + 1)
                else:
                    v += p * exp_value(state.getNextState(ghostIndex, action), d, ghostIndex + 1)
            return v

        res = [(action, exp_value(gameState.getNextState(0, action), 0, 1)) for action in gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])
        return res[-1][0]


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

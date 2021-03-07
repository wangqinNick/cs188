# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """

    class Node:
        def __init__(self, state, parent, action):
            self.state = state  # a tuple (x, y)
            self.parent = parent  # parent node
            self.action = action  # how to get to this state

    # frontier = {startNode}
    startNode = Node(state=problem.getStartState(),
                     parent=None,
                     action=None)

    frontier = util.Stack()
    frontier.push(startNode)

    # expanded = {}
    expanded = []
    solutions = []

    # while frontier is not empty:
    while not frontier.isEmpty():

        # node = frontier.pop()
        node = frontier.pop()

        # if isGoal(node):
        if problem.isGoalState(node.state):
            # return path_to_node
            solutions.append(node.action)
            while node.parent.action is not None:
                node = node.parent
                solutions.append(node.action)
            solutions.reverse()
            return solutions

        # if node not in expanded:
        if node.state not in expanded:
            # expanded.add(node)
            expanded.append(node.state)
            triples = problem.expand(node.state)
            for i in triples:
                child_state = i[0]
                child_action = i[1]
                child_cost = i[2]
                child_node = Node(state=child_state,
                                  parent=node,
                                  action=child_action)
                frontier.push(child_node)
    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    class Node:
        def __init__(self, state, parent, action):
            self.state = state  # a tuple (x, y)
            self.parent = parent  # parent node
            self.action = action  # how to get to this state

    # frontier = {startNode}
    startNode = Node(state=problem.getStartState(),
                     parent=None,
                     action=None)

    frontier = util.Queue()
    frontier.push(startNode)

    # expanded = {}
    expanded = []
    solutions = []

    # while frontier is not empty:
    while not frontier.isEmpty():

        # node = frontier.pop()
        node = frontier.pop()

        # if isGoal(node):
        if problem.isGoalState(node.state):
            # return path_to_node
            solutions.append(node.action)
            while node.parent.action is not None:
                node = node.parent
                solutions.append(node.action)
            solutions.reverse()
            return solutions

        # if node not in expanded:
        if node.state not in expanded:
            # expanded.add(node)
            expanded.append(node.state)
            triples = problem.expand(node.state)
            for i in triples:
                child_state = i[0]
                child_action = i[1]
                child_cost = i[2]
                child_node = Node(state=child_state,
                                  parent=node,
                                  action=child_action)
                frontier.push(child_node)
    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return util.manhattanDistance(state, problem.goal)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # print(heuristic(problem.getStartState(), problem=problem))

    class Node:
        def __init__(self, state, parent, action, cost):
            self.state = state  # a tuple (x, y)
            self.parent = parent  # parent node
            self.action = action  # how to get to this state
            self.cost = cost  # total (actual) cost to get to this node g(n)
            self.h = nullHeuristic(state, problem=problem)  # h(n)
            self.f = cost + self.h  # for any node, f(n) = g(n) + h(n)

    # frontier = {startNode}
    startNode = Node(state=problem.getStartState(),
                     parent=None,
                     action=None,
                     cost=0)

    frontier = util.PriorityQueue()
    frontier.push(startNode, priority=0)

    # expanded = {}
    expanded = []
    solutions = []

    # while frontier is not empty:
    while not frontier.isEmpty():

        # node = frontier.pop()
        node = frontier.pop()

        # if isGoal(node):
        if problem.isGoalState(node.state):
            # return path_to_node
            solutions.append(node.action)
            while node.parent.action is not None:
                node = node.parent
                solutions.append(node.action)
            solutions.reverse()
            return solutions

        # if node not in expanded:
        if node.state not in expanded:
            # expanded.add(node)
            expanded.append(node.state)
            triples = problem.expand(node.state)
            for i in triples:
                child_state = i[0]
                child_action = i[1]
                child_cost = i[2]
                child_node = Node(state=child_state,
                                  parent=node,
                                  action=child_action,
                                  cost=child_cost)
                frontier.push(child_node,
                              priority=child_node.f)
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch

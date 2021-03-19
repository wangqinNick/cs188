import numpy as np



# w: width of the board
# h: height of the board
# player_count: number of players (2 or 3)
# my_id: id of my player (0 = 1st player, 1 = 2nd player, ...)
w, h, player_count, my_id = [int(i) for i in input().split()]
depth = 8

def evaluationFunction(state):
    return 0  # Todo implement evaluation function

def solution(gameState, depth):
    GhostIndex = [i for i in range(1, gameState.getNumAgents())]

    def isTerminate(state, d):
        return state.isWin() or state.isLose() or d == depth

    def min_value(state, d, ghostIndex):  # minimizer

        if isTerminate(state, d):
            return evaluationFunction(state)

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
            return evaluationFunction(state)

        v = -float('inf')
        for action in state.getLegalActions(0):
            v = max(v, min_value(state.getNextState(0, action), d, 1))
        # print(v)
        return v

    res = [(action, min_value(gameState.getNextState(0, action), 0, 1)) for action in
           gameState.getLegalActions(0)]
    res.sort(key=lambda k: k[1])

    return res[-1][0]


# state: (x, y, walls_remain, grid)
def distance_to_goal(state):
    return w - state[1]


def utility(state):
    return distance_to_goal(state)


def getLegalActions(state):
    possible_directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    valid_actions_from_state = []
    for action in possible_directions:
        x, y = (state[0], state[1])
        dx, dy = action
        nextx, nexty = int(x + dx), int(y + dy)
        if walls[nextx][nexty] == 0:  # not a wall
            valid_actions_from_state.append(action)
    return valid_actions_from_state


def getNextState(state, action):
    x, y = state
    dx, dy = action
    nextx, nexty = int(x + dx), int(y + dy)
    return nextx, nexty, state[2], state[3]  # state: (x, y, walls_remain, grid)


def min_value(state, d):
    if d == depth:
        return utility(state)


# game loop
while True:
    for i in range(player_count):
        # x: x-coordinate of the player
        # y: y-coordinate of the player
        # walls_left: number of walls available for the player
        x, y, walls_left = [int(j) for j in input().split()]
    wall_count = int(input())  # number of walls on the board
    for i in range(wall_count):
        inputs = input().split()
        wall_x = int(inputs[0])  # x-coordinate of the wall
        wall_y = int(inputs[1])  # y-coordinate of the wall
        wall_orientation = inputs[2]  # wall orientation ('H' or 'V')
        if wall_orientation == 'H':
            for j in range(2):  # wall length = 2
                walls[wall_x + j][wall_y] = 1
        else:
            for j in range(2):  # wall length = 2
                walls[wall_x][wall_y + j] = 1
    # action: LEFT, RIGHT, UP, DOWN or "putX putY putOrientation" to place a wall
    print("RIGHT")

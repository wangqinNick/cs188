def aStarSearch():
    class Node:
        def __init__(self, state, parent, action, cost, prev_cost):
            self.state = state  # a tuple (x, y)
            self.parent = parent  # parent node
            self.action = action  # how to get to this state
            self.cost_sum = prev_cost + cost
            h = heuristic(state, problem=problem)  # h(n)
            self.f = self.cost_sum + h  # for any node, f(n) = g(n) + h(n)

    # frontier = {startNode}
    startNode = Node(state=problem.getStartState(),
                     parent=None,
                     action=None,
                     cost=0,
                     prev_cost=0)

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
                                  cost=child_cost,
                                  prev_cost=node.cost_sum)
                frontier.push(child_node,
                              priority=child_node.f)
    return None


# game loop
while True:
    x, y = [int(i) for i in input().split()]
    human_count = int(input())
    for i in range(human_count):
        human_id, human_x, human_y = [int(j) for j in input().split()]
    zombie_count = int(input())
    for i in range(zombie_count):
        zombie_id, zombie_x, zombie_y, zombie_xnext, zombie_ynext = [int(j) for j in input().split()]

    print("0 0")

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

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
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

def buildActions(node, parent):
    action = node[1]
    if action is None: return []
    return buildActions(parent[node], parent) + [action]

def graphSearchWithoutCosts(problem, fringe):

    # Dict to track who is the parent of each node
    # parent[node1] = node2 means that the parent of node1 is node2
    parent = dict()
    # Set of closed states (states that have already been visited)
    closed = set()

    # Insert starting node into the fringe
    # Note that a node is a tuple consisting of a state, an action and a cost
    # node[0] = state, node[1] = action, node[2] = cost
    start_state = problem.getStartState()
    start_action = None
    start_cost = 0
    start_node = (start_state, start_action, start_cost)
    fringe.push(start_node)

    while not fringe.isEmpty():
        node = fringe.pop()
        state = node[0]

        if problem.isGoalState(state): return buildActions(node, parent)

        if state not in closed:
            closed.add(state)

            for successor in problem.getSuccessors(state):
                successor_state = successor[0]

                if successor_state not in closed:
                    fringe.push(successor)
                    parent[successor] = node

    return []

def graphSearchWithCosts(problem, heuristic):

    # Dict to track who is the parent of each node
    # parent[node1] = node2 means that the parent of node1 is node2
    parent = {}
    # Set of closed states (states that have already been visited)
    closed = set()
    # Dict to track the cost of each node
    cost = {}
    fringe = util.PriorityQueue()

    # Insert starting node into the fringe
    # Note that a node is a tuple consisting of a state and an action
    # while the cost is the priority of this item
    # node[0] = state, node[1] = action, node[2] = cost
    start_state = problem.getStartState()
    start_action = None
    start_cost = 0
    fringe.push(item=(start_state, start_action), priority=start_cost)
    cost[start_state] = start_cost

    while not fringe.isEmpty():
        node = fringe.pop()
        state = node[0]

        if problem.isGoalState(state): return buildActions(node, parent)

        if state not in closed:
            closed.add(state)

            for successor in problem.getSuccessors(state):
                successor_state = successor[0]
                successor_cost = successor[2]

                if successor_state not in closed:
                    g = cost[state] + successor_cost
                    h = heuristic(successor_state, problem)
                    # To understand why this works, check how update function acts
                    fringe.update(successor[:2], g+h)
                    cost[successor_state] = g
                    parent[successor[:2]] = node

    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return graphSearchWithoutCosts(problem, util.Stack())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return graphSearchWithoutCosts(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return graphSearchWithCosts(problem, nullHeuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return graphSearchWithCosts(problem, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

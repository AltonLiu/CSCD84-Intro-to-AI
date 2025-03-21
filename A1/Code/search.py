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
from game import Directions
from typing import List

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


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    # Need to do this outside while loop because start node is a different format from successor nodes
    if problem.isGoalState(problem.getStartState()):
        return []
    
    frontier = util.Stack()
    for successor in problem.getSuccessors(problem.getStartState()):
        # Store the successor along with the path to it, that way backtracking is simplified (if we need to backtrack, the node we backtrack to will already have the corresponding path to it,
        # therefore we won't need to modify the path to accomodate for backtracking)
        frontier.push((successor, [successor[1]]))
    
    # Track which nodes have been visited so we visit each node at most 1 time
    visited = [problem.getStartState()]
    
    while not frontier.isEmpty():
        # Pop next node and its corresponding path from frontier
        (current_node, path) = frontier.pop()
        
        # DEBUG
        # print("current_node:")
        # print(current_node)
        # print("path:")
        # print(path)
        
        # No visited node will ever be added to the frontier, therefore current_node will always be unvisited
        visited.append(current_node[0])
        
        # If this node is a goal node return path from start to this node
        if problem.isGoalState(current_node[0]):
            # DEBUG
            # print("Path to goal:")
            # print(path)
            return path
        
        # Else add unvisited successor nodes to frontier
        for successor in problem.getSuccessors(current_node[0]):
            if successor[0] not in visited:
                # DEBUG
                # print("successor: ")
                # print(successor)
                frontier.push((successor, path + [successor[1]]))
        
    # Either goal was not found after exploring every node or the start node has no successors
    return []
    

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    pq = util.Queue()
    startNode = problem.getStartState()

    pq.push(startNode)
    actions = {startNode: []} # {node: [Directions], node: [Directions], ....}

    while not pq.isEmpty():
        node = pq.pop()
        if problem.isGoalState(node):
            return actions[node]

        for neighbour, action, _ in problem.getSuccessors(node):
            newPath = actions[node] + [action]
            
            if neighbour not in actions.keys():
                actions[neighbour] = newPath
                pq.push(neighbour)
                
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    # Need to do this outside while loop because start node is a different format from successor nodes
    if problem.isGoalState(problem.getStartState()):
        return []
    
    # Track which nodes have been visited so we visit each node at most 1 time
    # {coords: ([shortest_path], path_weight)}
    visited = {problem.getStartState(): ([], 0)}
    
    frontier = util.PriorityQueueWithFunction(statePriorityFunction)
    for successor in problem.getSuccessors(problem.getStartState()):
        # Store the successor along with the path to it and the total cost of the path to it, that way backtracking is simplified
        # (if we need to backtrack, the node we backtrack to will already have the corresponding path to it, therefore we won't need 
        # to modify the path to accomodate for backtracking)
        # (((x, y), direction_from_predecessor, cost_from_predecessor), [path], path_weight)
        visited[successor[0]] = ([successor[1]], successor[2])
        frontier.push((successor, [successor[1]], successor[2]))
    
    while not frontier.isEmpty():
        # Pop next node and its corresponding path from frontier
        current_node, path, path_weight = frontier.pop()
        
        # DEBUG
        # print("current_node:")
        # print(current_node)
        # print("path:")
        # print(path)
        
        current_coords = current_node[0]
        
        # If this node is a goal node return path from start to this node
        if problem.isGoalState(current_coords):
            return path
        
        # Else add unvisited successor nodes to frontier
        for successor in problem.getSuccessors(current_coords):
            successor_weight = successor[2]
            total_weight = path_weight + successor_weight
            if successor[0] not in visited or total_weight < visited[successor[0]][1]:
                # DEBUG
                # print("successor: ")
                # print(successor)
                # print("pushed:")
                # print((successor, path + [successor[1]], total_weight))
                visited[successor[0]] = (path + [successor[1]], total_weight)
                frontier.push((successor, path + [successor[1]], total_weight))
        
    # Either goal was not found after exploring every node or the start node has no successors
    return []

def statePriorityFunction(state):
    return state[2]

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    # not able to rediscover nodes
    pq = util.PriorityQueue()
    startNode = problem.getStartState()

    pq.push(startNode, 0)
    actions = {startNode: []} # {node: [Directions], node: [Directions], ....}

    while not pq.isEmpty():
        node = pq.pop()
        # finished.append(node)
        if problem.isGoalState(node):
            return actions[node]

        nodeWeight = problem.getCostOfActions(actions[node])
        for neighbour, action, stepCost in problem.getSuccessors(node):
            newPath = actions[node] + [action]
            
            if neighbour not in actions.keys():
                actions[neighbour] = newPath
                pq.push(neighbour, nodeWeight + stepCost + heuristic(neighbour, problem))
            else:
                oldPath = actions[neighbour]
                if (problem.getCostOfActions(newPath) >= problem.getCostOfActions(oldPath)):
                    continue
                actions[neighbour] = newPath
                pq.update(neighbour, problem.getCostOfActions(newPath) + heuristic(neighbour, problem))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

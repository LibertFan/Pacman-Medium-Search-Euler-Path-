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
from util import Stack
from util import Queue
from util import PriorityQueue

from game import Directions   
n = Directions.NORTH
s = Directions.SOUTH
e = Directions.EAST
w = Directions.WEST
o = Directions.STOP
    

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

def PeriodActionsFunc(ReverseState,StateParentDict):
  
    PeriodActions = []
    while True:
        ParentInfo = StateParentDict.get(ReverseState)
        ReverseState = ParentInfo[0]
        if ReverseState is None:
           break
        if ParentInfo[1] in [n,s,e,w]:
           PeriodActions.append(ParentInfo[1])
        else:     
           PeriodActions.extend(ParentInfo[1][-1::-1])
       
    return PeriodActions[-1::-1]


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    return  [s, s, w, s, w, w, s, w]

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
    CandidateStates = Stack()

    StartState = problem.getStartState() 
    CandidateStates.push(StartState)   
    ExploredStates = [StartState,]
    StateParentDict = {StartState:(None,o),}
    
    currentState = StartState
    while not CandidateStates.isEmpty():
          currentState = CandidateStates.pop()
          if problem.isGoalState(currentState):
             return PeriodActionsFunc(currentState,StateParentDict)
          else:
             for successor in problem.getSuccessors(currentState):  
                 state = successor[0]
                 direction = successor[1]
                 if state in ExploredStates:
                    continue
                 else:
                    CandidateStates.push(state)
                    StateParentDict[state] = (currentState,direction)   
                    ExploredStates.append(state) 
    
    return -1

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"     
    
    CandidateStates = Queue()  
    StartState = problem.getStartState() 
    CandidateStates.push(StartState)   
    ExploredStates = [StartState,]
    StateParentDict = {StartState:(None,o),} 
     
    while not CandidateStates.isEmpty():
          currentState = CandidateStates.pop()
          if problem.isGoalState(currentState):
             return PeriodActionsFunc(currentState,StateParentDict)
          else:
             for successor in problem.getSuccessors(currentState):
                 state = successor[0]
                 direction = successor[1]
                 if state in ExploredStates:
                    continue
                 else:
                    CandidateStates.push(state)
                    StateParentDict[state] = (currentState,direction)
                    ExploredStates.append(state)
    return -1

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    CandidateStates = PriorityQueue()

    StartState = problem.getStartState()
    CandidateStates.push(StartState,0)
    ExploredStates = []
    StateParentDict = {StartState:(None,o,0),}

    while not CandidateStates.isEmpty():
          currentState = CandidateStates.pop()
          ExploredStates.append(currentState)
          if problem.isGoalState(currentState):
             return PeriodActionsFunc(currentState,StateParentDict)
          else:
             for successor in problem.getSuccessors(currentState):
                 state = successor[0]
                 direction = successor[1]
                 cost = StateParentDict.get(currentState)[2] + successor[2]
                 if state in ExploredStates:
                    continue
                 else:
                    CandidateStates.update(state,cost)
                    StateParentDict[state] = (currentState,direction,cost)
		    
    return -1
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """    
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    CandidateStates = PriorityQueue()  
    StartState = problem.getStartState() 
    CandidateStates.push(StartState,0)   
    ExploredStates = []
    StateParentDict = {StartState:(None,o,0),}
    
    while not CandidateStates.isEmpty():
        currentState = CandidateStates.pop()
        ExploredStates.append(currentState)
        if problem.isGoalState(currentState):
            return PeriodActionsFunc(currentState,StateParentDict)
        else:
            for successor in problem.getSuccessors(currentState):
                 state = successor[0]
                 direction = successor[1]
                 if StateParentDict.get(currentState)[2]==0:
                     cost = StateParentDict.get(currentState)[2] + successor[2] + heuristic(state,problem)
                 else:
                     cost = StateParentDict.get(currentState)[2] + successor[2] + heuristic(state,problem) - problem.getHeuristicInfo(currentState)
                 if state in ExploredStates:
                    continue
                 else:
                    CandidateStates.update(state,cost)
                    StateParentDict[state] = (currentState,direction,cost)
                    if problem.getHeuristicInfo(currentState) - heuristic(state,problem) > successor[2]:
                       print "Heuristic function error:",state[0:2],heuristic(state,problem),problem.getHeuristicInfo(currentState),successor[2]

    return -1
    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

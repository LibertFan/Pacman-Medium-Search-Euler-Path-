# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import time
import search
import util
import collections
import copy

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs

    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        # if prob not in globals().keys() or not prob.endswith('Problem'):
        #    raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions = problem.path
        print self.actions
        #self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
        
    def getHeuristicInfo(self,state):
           return 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class Node(object):
    def __init__(self, Position):
        self.Position = Position
        self.LegalMoves = []

        # NeighNodeActionCost and successorNodeActionCost are all consist of 3-tuple 
        # which are respectively Next Node(class instance) and the shortest action 
        # from current node to next node and the cost of action(the length of the path).
        # NeighNodeActionCost is built for storing the basic information of graph, but 
        # successorNodeActionCost is special for the postprocessing.
        # successorNodeAction is aim at replacing the class attribution of 
        # NeighNodes, LegalActions, costs 
        self.NeighNodeActionCost = []
        self.SuccessorNodeActionCost = []
        self.NodeType = None

    def addLegalMove(self, MoveDirection):
        self.LegalMoves.append(MoveDirection)

    def isLegalMove(self, MoveDirection): 
        for NeighInfo in self.NeighNodeActionCost:
            Actions = NeighInfo[1]
            if Actions[0] == MoveDirection:
               return False
        return True

    # the following function is specified for node that only has two directions to move
    def getLegalExitMove(self, enterMove):
        for move in self.LegalMoves:
            if move != Directions.REVERSE.get(enterMove):
               return move
    # If 
    def isSingularity(self):
        SuccessorNum = 0
        for SuccessorInfo in self.SuccessorNodeActionCost:
            SuccessorNodePosition = SuccessorInfo[0].Position
            if self.Position == SuccessorNodePosition or len(SuccessorInfo[0].SuccessorNodeActionCost) == 1:
               continue
            SuccessorNum += 1     
        if SuccessorNum%2 == 0:
           return False
        else:
           return True

    def getNodeType(self,ExtraNodes):
        # return 1: DeadNode
        # return 2: PathNode
        # return 3: CrossNode or nodes need to be added to graph 
        if self.Position in ExtraNodes:
           self.NodeType = 3
        elif len(self.LegalMoves) < 0 or len(self.LegalMoves) > 4:
           raise Exception("This node is defined wrong!")
        elif len(self.LegalMoves) == 1:
           self.NodeType = 1
        elif len(self.LegalMoves) == 2:
           self.NodeType = 2
        else:
           self.NodeType = 3
        return self.NodeType

class MediumGraphicProblem:

    def __init__(self, startingGameState):
        #import EulerianCircuit
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState 
        self.startingPosition = startingGameState.getPacmanPosition()
        self._expand = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        # self.startState = self.startingPosition + (0,) * 4 
        self.costFn = lambda x: 1
        self.ExtraNodes = (self.startingPosition,)
        self.Graphic()
        self.ReGraphic()
        LengthOneEdgeList, Aggregates = self.Partition()
        K = 3
        allCrossNodes = connectGraph(LengthOneEdgeList, Aggregates)
        #print len(allCrossNodes)
        minCost = 99999
        minPath = []
        for crossNodes in allCrossNodes:
            self.Singular(crossNodes)
            EulerPath = self.ComputeBestEuler(crossNodes, K)
            
            eulerian = EulerianCircuit(self.PositionNodeDict.get(self.startingPosition), EulerPath[0])
            path, _, cost = eulerian.findCircuit()
            if cost < minCost:
               minCost = cost
               minPath = path
            
        #print self.PositionNodeDict.get(self.startingPosition)
        #print self.CrossNodes        
        self.path = minPath
        
    def Singular(self,CrossNodes):
        self.SingularPositions = []
        for CrossNode in CrossNodes:
            if len(CrossNode.SuccessorNodeActionCost)%2 != 0:
               self.SingularPositions.append(CrossNode.Position)  
        
    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

    def Graphic(self):
        from searchAgents import Node
        n = Directions.NORTH
        s = Directions.SOUTH
        e = Directions.EAST
        w = Directions.WEST
        # PositioNodeIDict reserve a map between Node Position and Node(class instance)
        PositionNodeDict = {}
        # CrossNodes is a list reserved Node that belongs the type-1,3
        CrossNodes = []

        for i in range(1, self.walls.width - 1):
            for j in range(1, self.walls.height - 1):
                if not self.walls[i][j]:
                   NewNode = Node((i,j))
                   for action in [n,s,e,w]:
                       dx,dy = Actions.directionToVector(action)
                       nextx, nexty = int(i + dx), int(j + dy)
                       if not self.walls[nextx][nexty]:
                          NewNode.addLegalMove(action)
                   try:
                      NewNodeType = NewNode.getNodeType(self.ExtraNodes)
                   except:
                      NewNodeType = NewNode.NodeType()
                   if NewNodeType != 2:
                      CrossNodes.append(NewNode)
                   PositionNodeDict[(i,j)] = NewNode

        for CrossNode in CrossNodes:
            CrossNodeX,CrossNodeY = CrossNode.Position
            for CrossLegalMove in CrossNode.LegalMoves:
                if not CrossNode.isLegalMove(CrossLegalMove):
                   continue
                LegalMove = CrossLegalMove
                dx,dy = Actions.directionToVector(LegalMove)
                NodePosition = (int(CrossNodeX + dx), int(CrossNodeY + dy))
                NextNode = PositionNodeDict.get(NodePosition)
                actions = [LegalMove,]
                cost = 1
                while NextNode.NodeType == 2:
                    LegalMove = NextNode.getLegalExitMove(LegalMove)
                    actions.append(LegalMove)
                    cost += 1
                    dx,dy = Actions.directionToVector(LegalMove)
                    x,y = NodePosition[0],NodePosition[1]
                    NodePosition = (int(x + dx),int(y + dy))
                    NextNode = PositionNodeDict.get(NodePosition)

                NextCrossNode = NextNode
                CrossNode.NeighNodeActionCost.append((NextCrossNode,actions,cost))
                InverseActions = []
                for direction in actions[-1::-1]:
                    InverseActions.append(Directions.REVERSE.get(direction))
                NextCrossNode.NeighNodeActionCost.append((CrossNode,InverseActions,cost))

        self.PositionNodeDict = PositionNodeDict
   
    # This function should partition the whole graph into several parts
    # through delete the edge whose length is just one and the vertex is not
    # a dead node. After such manipulation every part is not connected with any others.  
    def Partition(self):
        CrossNodes = copy.deepcopy(self.CrossNodes)

        LengthOneEdgeList = [] 
        for CrossNode in CrossNodes:
            if CrossNode.NodeType == 1:
               continue
            index = 0
            while index < len(CrossNode.SuccessorNodeActionCost):
                SuccessorInfo = CrossNode.SuccessorNodeActionCost[index] 
                SuccessorNode = SuccessorInfo[0]
                SuccessorCost = SuccessorInfo[2]
                if SuccessorCost == 1 and SuccessorNode.NodeType != 1:
                   CrossNode.SuccessorNodeActionCost.remove(SuccessorInfo)
                   Edge = {CrossNode.Position,SuccessorNode.Position}
                   if Edge not in LengthOneEdgeList:
                      LengthOneEdgeList.append(Edge)
                   continue
                index += 1  
                
        Positions = []
        PositionNodeDict = {}
        for CrossNode in CrossNodes:
            Positions.append(CrossNode.Position)
            PositionNodeDict[CrossNode.Position] = CrossNode            
       
        Aggregates = []
        while len(Positions) > 0:
            Position = Positions.pop()
            GNode1 = PositionNodeDict.get(Position)
            Aggregate = [GNode1,]
            index = 0 
            while index < len(Aggregate):
                CurrentNode = Aggregate[index]
                index += 1  
                for SuccessorInfo in CurrentNode.SuccessorNodeActionCost:
                    SuccessorNode = SuccessorInfo[0]
                    if SuccessorNode not in Aggregate:   
                       Aggregate.append(SuccessorNode)
                       Positions.remove(SuccessorNode.Position)  
            Aggregates.append(Aggregate)
        return LengthOneEdgeList, Aggregates        
    
    def ReGraphic(self):
        DeadNodes = []
        CrossNodes = []
        for Position,GNode in self.PositionNodeDict.items():
            if GNode.NodeType == 1:
               DeadNodes.append(GNode)
            elif GNode.NodeType == 3:
               CrossNodes.append(GNode)
   
        for DeadNode in DeadNodes:
            NextCrossNodeInfo = DeadNode.NeighNodeActionCost[0]
            DeadNode.NeighNodeActionCost.append(NextCrossNodeInfo)
            DeadNode.SuccessorNodeActionCost = DeadNode.NeighNodeActionCost[:]
            NextCrossNode = NextCrossNodeInfo[0]
            for NeighInfo in NextCrossNode.NeighNodeActionCost:
                if NeighInfo[0] == DeadNode:
                   NextCrossNode.NeighNodeActionCost.append(NeighInfo)
                   break

        for CrossNode in CrossNodes:
            CrossNode.SuccessorNodeActionCost = CrossNode.NeighNodeActionCost[:]
        # return is list Node Class !
        self.DeadNodes = DeadNodes[:]
        self.CrossNodes = CrossNodes[:] + DeadNodes[:]               
        # The NeighNode is the Successor Node
        SingularPositions = []
        for CrossNode in CrossNodes:
            if CrossNode.isSingularity():
               SingularPositions.append(CrossNode.Position)
        # return self.SingularPositions is list of Position
        self.SingularPositions = SingularPositions[:]

    def ComputeBestConnect(self, Nodes, Cost, K, optimalbound):
        '''
        Given Nodes and Cost, compute best connect choice. The optimalbound is bound of solutions before
        '''
        EulerCsp = create_Euler_csp(Nodes, Cost)
        search = BacktrackingSearch()
        search.solve(EulerCsp, optimalbound)
        return [search.optimalAssignment,search.optimalWeight,search.numOptimalAssignments]

    def ComputeBestEuler(self, Crossnodes, K):
        '''
        Best Result ia saved in a 3-item list EulerBest, the first item is list of optimal assignment, the second is the optimal cost
        the third is the number of optimal assignment
        '''
        Cost_single = self.ComputeMinDistance(self.SingularPositions, self.SingularPositions, K)
        Cost_all = Cost_single #save all cost in Cost_all
        ## print self.SingularPositions
        ## Case1 not consider starting point
        ## print 'case1'
        EulerBest = self.ComputeBestConnect(self.SingularPositions, Cost_single, K, 999999)
        ## print EulerBest
        ## Compute starting point information
        Cost_temp = self.ComputeMinDistance([self.startingPosition,], self.SingularPositions, K)
        Cost_start = Cost_temp
        for (p1,p2),value in Cost_temp.items():
            Cost_start[(p2,p1)] = value
        Cost_all.update(Cost_start)
        StartNeighbor = []
        for position in self.SingularPositions:
            try:
                if Cost_start.get((self.startingPosition, position)) is not None:
                   StartNeighbor.append(position)
            except ValueError:
                pass   
        ## Case2 connect starting point with one singular point(position1) and end in another singluar point(position2)
        ## print 'case2'
        for position1 in StartNeighbor:
            for position2 in self.SingularPositions:
                Positions = self.SingularPositions[:]
                try:  
                  Positions.remove(position1)
                  Positions.remove(position2)
                except ValueError:
                  pass
                  
                EulerCurrent = self.ComputeBestConnect(Positions, Cost_single, K, EulerBest[1] - Cost_start[(self.startingPosition, position1)][1])
                EulerCurrent[1] += Cost_start[(self.startingPosition, position1)][1]
                ## update EulerBest
                if EulerCurrent[1] <= EulerBest[1]:
                    for i in range(EulerCurrent[2]):
                        EulerCurrent[0][i][self.startingPosition] = position1
                        EulerCurrent[0][i][position1] = self.startingPosition
                    if EulerCurrent[1] < EulerBest[1]:
                        EulerBest = EulerCurrent
                    else:
                        EulerBest[0] = EulerBest[0]+EulerCurrent[0]
                        EulerBest[2] += EulerCurrent[2] 
        ## print EulerBest
        ## Case3 connect starting point with one singular point(position1) and end in deadend(Node1) 
        ## print 'case3'       
        DeadNodesNeigh = []
        for Node1 in self.DeadNodes:
            position2 = Node1.NeighNodeActionCost[0][0].Position
            
            ## delete position2 
            if position2 in self.SingularPositions:
                for position3 in StartNeighbor:
                    Positions = self.SingularPositions[:]
                    Positions.remove(position2)
                    Positions.remove(position3)
                    EulerCurrent = self.ComputeBestConnect(Positions, Cost_single, K, EulerBest[1] - Cost_start[(self.startingPosition, position3)][1] + Node1.NeighNodeActionCost[0][2])
                    EulerCurrent[1] += Cost_start[(self.startingPosition, position3)][1] 
                    EulerCurrent[1] -= Node1.NeighNodeActionCost[0][2]
                    ## update EulerBest
                    if EulerCurrent[1] <= EulerBest[1]:
                        DeadNodesNeigh.append(position2)
                        for i in range(EulerCurrent[2]):
                            EulerCurrent[0][i][self.startingPosition] = position3
                            EulerCurrent[0][i][position3] = self.startingPosition
                        if EulerCurrent[1] < EulerBest[1]:
                            EulerBest = EulerCurrent
                        else:
                            EulerBest[0] = EulerBest[0]+EulerCurrent[0]
                            EulerBest[2] += EulerCurrent[2]                    
            else:
            ## add position2
                for position3 in StartNeighbor:
                    Positions = self.SingularPositions[:]
                    Positions.remove(position3)
                    Positions.append(position2)
                    Cost_temp = self.ComputeMinDistance([position2,], Positions, K)
                    Cost_2 = Cost_temp
                    for (p1,p2),value in Cost_temp.items():
                        Cost_2[(p2,p1)] = value
                    Cost_all.update(Cost_2)
                    Cost_2.update(Cost_single)
                    EulerCurrent = self.ComputeBestConnect(Positions, Cost_2, K, EulerBest[1] - Cost_start[(self.startingPosition, position3)][1] + Node1.NeighNodeActionCost[0][2])
                    EulerCurrent[1] += Cost_start[(self.startingPosition, position3)][1]
                    EulerCurrent[1] -= Node1.NeighNodeActionCost[0][2]
                    ## update EulerBest
                    if EulerCurrent[1] <= EulerBest[1]:
                        DeadNodesNeigh.append(position2)
                        for i in range(EulerCurrent[2]):
                            EulerCurrent[0][i][self.startingPosition] = position3
                            EulerCurrent[0][i][position3] = self.startingPosition
                        if EulerCurrent[1] < EulerBest[1]:
                            EulerBest = EulerCurrent
                        else:
                            EulerBest[0] = EulerBest[0]+EulerCurrent[0]
                            EulerBest[2] += EulerCurrent[2]
        ## change Crossnodes
        ## print EulerBest
        EulerOptimalCrossnodes = [] 
        for i in range(EulerBest[2]):
            EulerCrossnodes = copy.deepcopy(Crossnodes)
            PositionNodeDict = dict()
            for CrossNode in EulerCrossnodes:
                PositionNodeDict[CrossNode.Position] = CrossNode
            flag = 0    
            for position1, position2 in EulerBest[0][i].items():
                node1, node2 = PositionNodeDict[position1], PositionNodeDict[position2]
                if flag == 0: 
                   if position1 in DeadNodesNeigh:
                      for SuccessorInfo in node1.SuccessorNodeActionCost:
                          SuccessorNode = SuccessorInfo[0]
                          if len(SuccessorNode.NeighNodeActionCost) == 1:
                             node1.SuccessorNodeActionCost.remove(SuccessorInfo)
                             flag = 1
                             break                             
                EulerCrossnodes[EulerCrossnodes.index(node1)].SuccessorNodeActionCost.append((node2,Cost_all[(position1,position2)][0],Cost_all[(position1,position2)][1]))
            EulerOptimalCrossnodes.append(EulerCrossnodes)

        return EulerOptimalCrossnodes
   
    def ComputeMinDistance(self,GNodes1,GNodes2,K):  
        if K > len(GNodes2) - 1:
           print "K is larger than the maximum number of neighbour nodes"
           K = len(GNodes2) - 1
        NNActionCostDict = dict()
        for GNode1 in GNodes1:
            NodeActionCostList = []
            for GNode2 in GNodes2:
                if GNode1 != GNode2:  
                   actions, cost =  mazeDistance(GNode1,GNode2,self.startingGameState)
                   NodeActionCostList.append((GNode2,actions,cost))
            NodeActionCostList = sorted(NodeActionCostList,key=lambda NodeActionCostList: NodeActionCostList[2])   
            for Info in NodeActionCostList[:K]:
                NNActionCostDict[(GNode1,Info[0])] = (Info[1],Info[2]) 
        return NNActionCostDict

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    actions = search.bfs(prob)
    return actions, len(actions)

class EulerianCircuit: 
    def __init__(self, startNode, nodeList):
        self.startPosition = startNode.Position
        for node in nodeList:
            if node.Position == self.startPosition:
               self.startNode = node 
        self.nodeList = nodeList
        self.nodeStack = util.Stack()
        self.nodesAccessList = self.getNodesAccessList()
    
    def getNodesAccessList(self):
        nodesAccessList = []
        for thisNode in self.nodeList:
            nodeAccess = [0,]*len(self.nodeList)
            for accessNodeActionCost in thisNode.SuccessorNodeActionCost:
                accessNode = accessNodeActionCost[0]
                i = self.nodeList.index(accessNode)
                nodeAccess[i] = nodeAccess[i] + 1
            nodesAccessList.append(nodeAccess)
        return nodesAccessList 
    
    def visit(self, node):
        u = self.nodeList.index(node)
        for accessNodeActionCost in node.SuccessorNodeActionCost:
            accessNode = accessNodeActionCost[0]
            v = self.nodeList.index(accessNode)
            if self.nodesAccessList[u][v]>0:
                self.nodesAccessList[u][v] = self.nodesAccessList[u][v]-1
                self.nodesAccessList[v][u] = self.nodesAccessList[v][u]-1
                self.visit(accessNode)
        self.nodeStack.push(node)        
        
    def findCircuit(self):
        from game import Directions
        self.visit(self.startNode)
        pathNodes = []
        while not self.nodeStack.isEmpty():
            pathNodes.append(self.nodeStack.pop())
        
        path = []
        pathCost = []
        helpDict = {}
        for i in range(len(pathNodes)-1):
            if helpDict.has_key((pathNodes[i],pathNodes[i+1])):
                helpDict[(pathNodes[i],pathNodes[i+1])] = helpDict[(pathNodes[i],pathNodes[i+1])] + 1
                helpDict[(pathNodes[i+1],pathNodes[i])] = helpDict[(pathNodes[i+1],pathNodes[i])] + 1      
            else:
                helpDict[(pathNodes[i],pathNodes[i+1])] = 1
                helpDict[(pathNodes[i+1],pathNodes[i])] = 1 

        for i in range(len(pathNodes)-1):
            count = 0
            for accessNode in pathNodes[i].SuccessorNodeActionCost:
                if accessNode[0] == pathNodes[i+1]:
                    count = count+1
                    if count == helpDict[(pathNodes[i],pathNodes[i+1])]:
                        path = path + accessNode[1]
                        pathCost.append(accessNode[2])
                        helpDict[(pathNodes[i],pathNodes[i+1])] = helpDict[(pathNodes[i],pathNodes[i+1])] - 1
                        helpDict[(pathNodes[i+1],pathNodes[i])] = helpDict[(pathNodes[i+1],pathNodes[i])] - 1  
                        break
           
        pathList = []
        index = 0 
        for cost in pathCost:
            pathList.append(path[index:(index+cost)]) 
            index += cost    
        PositionTimeDict = {}
        for GNode in self.nodeList:
            PositionTimeDict[GNode.Position] = (len(GNode.SuccessorNodeActionCost),len(GNode.SuccessorNodeActionCost))
       
        AllTime, PathTime = PositionTimeDict[self.startNode.Position]
        PositionTimeDict[self.startNode.Position] = (AllTime, PathTime-1)

        for i,PNode in enumerate(pathNodes[1:]):
            Position = PNode.Position
            AllTime, PathTime = PositionTimeDict[Position]
            PositionTimeDict[Position] = (AllTime,PathTime-1) 
            if PathTime == 1:
               if AllTime > 1:
                  pathList[i] = pathList[i][:-1]
                  PositionTimeDict[Position] = (AllTime,PathTime-1)  
            elif AllTime != 2:
               action1 = pathList[i][-1]
               action2 = pathList[i+1][0]
               if action1 == Directions.REVERSE[action2]:
                  pathList[i] = pathList[i][:-1] 
                  pathList[i+1] = pathList[i+1][1:]
               PositionTimeDict[Position] = (AllTime,PathTime-2) 
            else:
               PositionTimeDict[Position] = (AllTime,PathTime-2) 
        
        path = []
        for val in pathList:
            path.extend(val) 
        
        Cost = len(path)
        
        return [path,pathNodes,Cost]

class CSP:
    def __init__(self):
        # Total number of variables in the CSP.
        self.numVars = 0

        # The list of variable names in the same order as they are added. A
        # variable name can be any hashable objects, for example: int, str,
        # or any tuple with hashtable objects.
        self.variables = []

        # Each key K in this dictionary is a variable name.
        # values[K] is the list of domain values that variable K can take on.
        self.values = {}

        # This dict saves lowerbound for each variable and 'sum' is the sum of all variables
        self.lowerbound = {}

        # Each entry is a unary factor table for the corresponding variable.
        # The factor table corresponds to the weight distribution of a variable
        # for all added unary factor functions. If there's no unary function for 
        # a variable K, there will be no entry for K in unaryFactors.
        # E.g. if B \in ['a', 'b'] is a variable, and we added two
        # unary factor functions f1, f2 for B,
        # then unaryFactors[B]['a'] == f1('a') * f2('a')

        self.unaryFactors = {}

        # Each entry is a dictionary keyed by the name of the other variable
        # involved. The value is a binary factor table, where each table
        # stores the factor value for all possible combinations of
        # the domains of the two variables for all added binary factor
        # functions. The table is represented as a dictionary of dictionary.
        #
        # As an example, if we only have two variables
        # A \in ['b', 'c'],  B \in ['a', 'b']
        # and we've added two binary functions f1(A,B) and f2(A,B) to the CSP,
        # then binaryFactors[A][B]['b']['a'] == f1('b','a') * f2('b','a').
        # binaryFactors[A][A] should return a key error since a variable
        # shouldn't have a binary factor table with itself.

        self.binaryFactors = {}

    def add_variable(self, var, domain):
        """
        Add a new variable to the CSP.
        """
        if var in self.variables:
            self.values[var].append(domain)
        else:
            self.numVars += 1
            self.variables.append(var)
            self.values[var] = [domain]
            self.unaryFactors[var] = None
            self.binaryFactors[var] = dict()

    def get_neighbor_vars(self, var):
        """
        Returns a list of variables which are neighbors of |var|.
        """
        return self.binaryFactors[var].keys()

    def add_unary_factor(self, var, factorFunc):
        """
        Add a unary factor function for a variable. Its factor
        value across the domain will be *merged* with any previously added
        unary factor functions through elementwise multiplication.

        How to get unary factor value given a variable |var| and
        value |val|?
        => csp.unaryFactors[var][val]
        """
        factor = {val:float(factorFunc(val)) for val in self.values[var]}
        if self.unaryFactors[var] is not None:
            assert len(self.unaryFactors[var]) == len(factor)
            self.unaryFactors[var] = {val:self.unaryFactors[var][val] * \
                factor[val] for val in factor}
        else:
            self.unaryFactors[var] = factor

    def add_binary_factor(self, var1, var2, factor_func, Cost, lowerbound):
        """
        Takes two variable names and a binary factor function
        |factorFunc|, add to binaryFactors. If the two variables already
        had binaryFactors added earlier, they will be *merged* through element
        wise multiplication.

        How to get binary factor value given a variable |var1| with value |val1| 
        and variable |var2| with value |val2|?
        => csp.binaryFactors[var1][var2][val1][val2]
        """
        # never shall a binary factor be added over a single variable
        try:
            assert var1 != var2
        except:
            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            print '!! Tip:                                                                       !!'
            print '!! You are adding a binary factor over a same variable...                     !!'
            print '!! Please check your code and avoid doing this.                               !!'
            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            raise

        self.update_binary_factor_table(var1, var2,
            {val1: {val2: float(factor_func(var1, var2, val1, val2, Cost, lowerbound)) \
                for val2 in self.values[var2]} for val1 in self.values[var1]})
        self.update_binary_factor_table(var2, var1, \
            {val2: {val1: float(factor_func(var2, var1, val2, val1, Cost, lowerbound)) \
                for val1 in self.values[var1]} for val2 in self.values[var2]})

    def update_binary_factor_table(self, var1, var2, table):
        """
        Private method you can skip for 0c, might be useful for 1c though.
        Update the binary factor table for binaryFactors[var1][var2].
        If it exists, element-wise multiplications will be performed to merge
        them together.
        """
        if var2 not in self.binaryFactors[var1]:
            self.binaryFactors[var1][var2] = table
        else:
            currentTable = self.binaryFactors[var1][var2]
            for i in table:
                for j in table[i]:
                    assert i in currentTable and j in currentTable[i]
                    currentTable[i][j] *= table[i][j]

class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP solver. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = []

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param var: name of an unassigned variable.
        @param val: the proposed value.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert var not in assignment
        if val not in assignment:
            return (self.csp.binaryFactors[var][val][val][var]+self.csp.binaryFactors[val][var][var][val])/2
        else:
            return 0

    def solve(self, csp, optimalbound):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param ac3: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Reset solutions from previous search.
        self.reset_results()
        self.optimalWeight = optimalbound - self.csp.lowerbound['sum']

        # The dictionary of domains of every variable in the CSP.
        self.domains = {
            var: list(self.csp.values[var]) for var in self.csp.variables}
        
        # Perform backtracking search.
        self.backtrack({}, 0, 0)

        # Change optimalWeight
        if self.optimalAssignment:
            self.optimalWeight += self.csp.lowerbound['sum']
        else:
            self.optimalWeight = 999999

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """
        
        self.numOperations += 1
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            newAssignment = {}
            for var in self.csp.variables:
                newAssignment[var] = assignment[var]

            if len(self.optimalAssignment) == 0 or weight <= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                    self.optimalAssignment.append(newAssignment)
                else:
                    self.numOptimalAssignments = 1
                    self.optimalAssignment = [newAssignment]
                self.optimalWeight = weight
            return
        # Early Stop
        if weight >= self.optimalWeight:
            return
        # Select the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)
        # Get an ordering of the values.
        ordered_values = self.domains[var]
        # Continue the backtracking recursion using |var| and |ordered_values|.
        
        # Arc consistency check is enabled.
        for val in ordered_values:
            deltaWeight = self.get_delta_weight(assignment, var, val)
            if deltaWeight > 0:
                assignment[var] = val
                assignment[val] = var
                # create a deep copy of domains as we are going to look
                # ahead and change domain values
                localCopy = copy.deepcopy(self.domains)
                # fix value for the selected variable so that hopefully we
                # can eliminate values for other variables
                self.domains[var] = [val]
                self.domains[val] = [var]

                # enforce arc consistency
                self.arc_consistency_check(assignment,var,val)

                self.backtrack(assignment, numAssigned +
                               2, weight + deltaWeight)
                # restore the previous domains
                self.domains = localCopy
                del assignment[var]
                del assignment[val]

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return a currently unassigned variable.

        @param assignment: A dictionary of current assignment. This is the same as
            what you've seen so far.

        @return var: a currently unassigned variable.
        """
        count_min = 999999
        var_min = None
        for var in self.csp.variables:
            count = 0
            if var in assignment:
                continue
            else:
                for val in self.domains[var]:
                    if self.get_delta_weight(assignment, var, val) != 0:
                        count += 1
                if count < count_min:
                    count_min = count
                    var_min = var
        return var_min


    def arc_consistency_check(self, assignment, var1, var2):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.
        """
        
        def Revise(var1, var2, assignment):
            #change domain of var1
            revised = False
            d1 = self.domains[var1][:]
            d2 = self.domains[var2][:]
            for val1 in d1:
                remove = True
                for val2 in d2:
                    if self.csp.binaryFactors[var1][var2][val1][val2] != 0 or val1 not in assignment:
                        remove = False 
                        break
                if remove:
                    self.domains[var1].remove(val1)
                    revised = True
            return revised
        
        ac_queue = []
        for var in self.csp.get_neighbor_vars(var1):
            if var not in assignment:
                ac_queue.append([var,var1])
        for var in self.csp.get_neighbor_vars(var2):
            if var not in assignment:
                ac_queue.append([var,var2])
        while ac_queue:
            [var_1, var_2] = ac_queue.pop()
            if Revise(var_1, var_2, assignment):
                for var in self.csp.get_neighbor_vars(var_1):
                    if var != var_2  and var not in assignment:
                        ac_queue.append([var, var_1])
        
def create_Euler_csp(SingularNodes, Cost):
    '''
    create a Euler path csp problem
    '''
    csp = CSP()
    def are_neighbors(Node1, Node2):
        try:
            return Cost[(Node1,Node2)][1] == Cost[(Node2,Node1)][1]
        except:
            return False
    def connect_factor(a, b, c, d, Cost, lowerbound):
        if a == d and b == c:
            return Cost[(a,b)][1]-lowerbound[a]
        else:
            return 0
    ## add variables of csp
    for i in range(len(SingularNodes)):
        for j in range(i+1, len(SingularNodes)):
            Node1 = SingularNodes[i]
            Node2 = SingularNodes[j]
            if are_neighbors(Node1, Node2):
                csp.add_variable(Node1, Node2)
                csp.add_variable(Node2, Node1)
    ## compute lowerbound of every variable and the sum of them
    csp.lowerbound['sum'] = 0

    for var1 in csp.variables:
        for var2 in csp.variables:
            if are_neighbors(var1, var2):
                try:
                    csp.lowerbound[var1] = min(csp.lowerbound[var1],(Cost[(var1,var2)][1]-1))
                except:
                    csp.lowerbound[var1] = Cost[(var1,var2)][1]-1                
        csp.lowerbound['sum'] += csp.lowerbound[var1]/2.0
    ## add binary factor of csp
    for i in range(len(SingularNodes)):
        for j in range(i+1, len(SingularNodes)):
            Node1 = SingularNodes[i]
            Node2 = SingularNodes[j]
            if are_neighbors(Node1, Node2):
                csp.add_binary_factor(Node1, Node2, connect_factor, Cost, csp.lowerbound)
    return csp

def contrastAction(action):
    if action == 'North':
        return 'South'
    elif action == 'South':
        return 'North'
    elif action == 'West':
        return 'East'
    elif action == 'East':
        return 'West'   
    
def inWhichSeparete(nodePosition,separeteCrossNodes):
    for which,separete in enumerate(separeteCrossNodes):
        has = False
        for node in separete:
            if nodePosition == node.Position:
                has = True
                break
        if has:
            return which
            
def getIndexNode(nodeList1, nodeList2, nodesPosSet):
    returnNode = []
    nodesPosition = [i for i in nodesPosSet]
    for node in nodeList1:
        if nodesPosition[0] == node.Position:
            returnNode.append(node)
            break
        elif nodesPosition[1] == node.Position:
            returnNode.append(node)
            break
        
    for node in nodeList2:
        if nodesPosition[0] == node.Position:
            returnNode.append(node)
            break
        elif nodesPosition[1] == node.Position:
            returnNode.append(node)
            break
    return returnNode
   
def generateOriginRecord(separeteNum):
    listRecord = []
    for i in range(separeteNum):
        listRecord.append([i])
    return listRecord
        
def addToWhich(listRecord, ith):
    for i,record in enumerate(listRecord):
        if ith in record:
            return i
            
def isConnected(listRecord):
    num = 0
    for record in listRecord:
        if len(record)>0:
            num = num + 1
    if num == 1:
        return True
    else:
        return False
    
def connectGraph(waveLineList, separeteCrossNodes):
    allCrossNodes = []
    import itertools
    from game import Directions
    waveLineDict = {}
    
    for waveline in waveLineList:
        wavelineTuple = tuple([i for i in waveline])
        x = inWhichSeparete(wavelineTuple[0],separeteCrossNodes)
        y = inWhichSeparete(wavelineTuple[1],separeteCrossNodes)
        if x<y:
            waveLineDict[wavelineTuple] = (x,y)
        elif y<x:
            waveLineDict[wavelineTuple] = (y,x)
    lineNum = len(waveLineList)
    connectNum = len(separeteCrossNodes)
    chooseStr = ''
    for ith in range(lineNum):
        chooseStr = chooseStr + str(ith)
    chooseList = list(itertools.combinations(chooseStr,connectNum-1))
    inw = 0
    for choose in chooseList:
        inw = inw+1
        tryCrossNodes = copy.deepcopy(separeteCrossNodes)
        listRecord = generateOriginRecord(len(tryCrossNodes))
        for lineNum in choose:
            position = waveLineDict.get(tuple([i for i in waveLineList[int(lineNum)]]))
            if position is None:
                continue
            else:
                (x,y) = position
                twoNodes = getIndexNode(tryCrossNodes[x], tryCrossNodes[y], waveLineList[int(lineNum)])
                node0 = twoNodes[0]
                node1 = twoNodes[1]
                for neigh in node0.NeighNodeActionCost:
                    if neigh[0].Position == node1.Position:
                        node0.SuccessorNodeActionCost.append((node1,neigh[1],1))
                        node1.SuccessorNodeActionCost.append((node0,[Directions.REVERSE[neigh[1][0]]],1))
                        break
                if len(listRecord[x]) > 0:
                    if len(listRecord[y])>0:
                        listRecord[x] = listRecord[x] + listRecord[y]
                        listRecord[y] = []
                    else:
                        record = addToWhich(listRecord, y)
                        if not record == x:
                            listRecord[x] = listRecord[x] + listRecord[record]
                            listRecord[record] = []
                else:
                    if len(listRecord[y])>0: 
                        record1 = addToWhich(listRecord, x)
                        if not record1 == y:
                            listRecord[record1] = listRecord[record1] + listRecord[y]
                            listRecord[y] = []
                    else:
                        record1 = addToWhich(listRecord, x)
                        record2 = addToWhich(listRecord, y)
                        if not record1 == record2:
                            listRecord[record1] = listRecord[record1] + listRecord[record2]
                            listRecord[record2] = []
        if isConnected(listRecord):
            crossNodes = []
            for separate in tryCrossNodes:
                crossNodes = crossNodes + separate
            allCrossNodes.append(crossNodes)
            
    return allCrossNodes

    

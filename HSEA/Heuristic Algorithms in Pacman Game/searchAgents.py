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
A* search , run the following command:

> python pacman.py -p SearchAgent -a fn=astar

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
from pacman import GameState
import util
import time
import search


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='astar', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)
        #self.actionIndex = 0

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
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
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def getGoalState(self):
        return self.goal

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


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState:GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost
    # def getGhostPositions(self):
    #     return self.g

    # def getGhostStates())
class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)#输入问题返回actions
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """

    "*** YOUR CODE HERE ***"
    """
    admissible：到目标的h不会超过实际开销
    position:当前位置
    foodGrid:食物矩阵，T为食物
    foodGrid.asList():[(2, 1), (3, 3), (4, 1), (4, 2)]食物坐标
    problem.walls:墙矩阵，F为可通行区域
    problem.walls.count():总计有多少坐标是墙
    foodGrid._cellIndexToPosition(9)：数字转为坐标
    """
    position, foodGrid = state


    ######################################################################
    # 任务二
    ######################################################################

    # 原始
    # return 0
    #尝试1：到最远食物的曼哈顿距离
    # lst=list(map(lambda x: util.manhattanDistance(position,x), foodGrid.asList()))
    # if len(lst)>0:
    #     a=max(lst)
    #     print(a)
    #     return a
    # else:
    #     return 0
    #尝试2：到所有残存食物曼哈顿距离总和/总食物数
    # init_food=0
    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # init_food=max(init_food,len(lst))
    # if len(lst) > 0:
    #     a=sum(lst)/init_food
    #     print(a)
    #     return a
    # else:
    #     return 0
    # #尝试3：到所有残存食物曼哈顿距离总和/(总食物数/2)
    # init_food=0
    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # init_food=max(init_food,len(lst))
    # if len(lst) > 0:
    #     a=sum(lst)*2/init_food
    #     print(a)
    #     return a
    # else:
    #     return 0
    # #尝试4：到所有残存食物曼哈顿距离总和/残存食物数
    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # if len(lst) > 0:
    #     a=sum(lst)/len(lst)
    #     print(a)
    #     return a
    # else:
    #     return 0
    # #尝试5：到所有残存食物曼哈顿距离总和/(残存食物数/2)
    # init_food=0
    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # init_food=max(init_food,len(lst))
    # if len(lst) > 0:
    #     a=sum(lst)*2/len(lst)
    #     print(a)
    #     return a
    # else:
    #     return 0
    # #尝试6：到所有残存食物曼哈顿距离总和
    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # if len(lst) > 0:
    #     a=sum(lst)
    #     print(a)
    #     return a
    # else:
    #     return 0
    #尝试7：剩余食物数量
    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # if len(lst) > 0:
    #     a=len(lst)
    #     print(a)
    #     return a
    # else:
    #     return 0
    #

    # lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    # if len(lst) > 0:
    #     a=len(lst)+min(lst)
    #     #print(a)
    #     print(position)
    #     return a
    # else:
    #     return 0


    ######################################################################
    #任务三
    ######################################################################
    """
    state.getGhostPositions()
    state.getGhostStates()
    """
    # print(foodGrid)
    # print()
    # print(GameState.getGhostStates(problem))
    # print(GameState.getGhostPositions(problem))
    # print(state.getGhostPositions())
    # print(state.getGhostStates())
    lst = list(map(lambda x: util.manhattanDistance(position, x), foodGrid.asList()))
    if len(lst) > 0:
        a = len(lst) + min(lst)
        # print(a)
        print(len(lst),position)
        return a
    else:
        return 0



"""
python pacman.py -l Search1 -p AStarFoodSearchAgent
python pacman.py -l Search2 -p AStarFoodSearchAgent
python pacman.py -l Search3 -p AStarFoodSearchAgent
"""
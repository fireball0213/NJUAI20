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

class SearchProblem:#被继承
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
    return  [s, s, w, s, w, w, s, w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial平凡的.
    """
    return 0

def myHeuristic(state, problem=None):
    """
        you may need code other Heuristic function to replace  NullHeuristic
        """
    "*** YOUR CODE HERE ***"
    #采用封装好的曼哈顿距离
    return util.manhattanDistance(state, problem.getGoalState())#自己添加的getGoalState()#549.535，53
    #return util.EuclideanDistance( state, problem.getGoalState() )#557，550，56



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first.

        Your search algorithm needs to return a list of actions that reaches the
        goal. Make sure to implement a graph search algorithm.

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print("Start:", problem.getStartState())
        print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        print("Start's successors:", problem.getSuccessors(problem.getStartState()))
        """
    "*** YOUR CODE HERE ***"



    #A*自定义g+h
    def astar_priorityFunction(item):
        state, actions = item
        g = problem.getCostOfActions(actions)
        h = heuristic(state, problem)
        return g + h

    # 初始即目标
    if problem.isGoalState(problem.getStartState()):
        return []

    actions=[]
    visited=[]
    stateQ= util.PriorityQueueWithFunction(astar_priorityFunction)#只传函数名，不传参，否则会出现，int is not callable奇怪的错误
    stateQ.push(item=(problem.getStartState(),actions))

    while stateQ.isEmpty()==False:
        nowstate,nowactions=stateQ.pop()
        if problem.isGoalState(nowstate):
            return nowactions
        if nowstate not in visited:
            visited.append(nowstate)
            nowsuccessors=problem.getSuccessors(nowstate)
            for nextstate,action,cost in nowsuccessors:
                stateQ.push((nextstate,nowactions+[action]))
    return actions

    #util.raiseNotDefined()


# Abbreviations缩写
astar = aStarSearch

"""
python pacman.py -l smallMaze -p SearchAgent -a fn=astar,heuristic=myHeuristic
python pacman.py -l openMaze -p SearchAgent -a fn=astar,heuristic=myHeuristic
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=myHeuristic
坐标规则，左下角是（1，1）
"""
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        for ghost in newGhostStates:
            d = manhattanDistance(newPos, ghost.getPosition())
            if d <= 2:
                if ghost.scaredTimer > 10:
                    score += 2000.0
                else: 
                    score -= 200

        for food in newFood.asList():
            d = manhattanDistance(newPos, food);
            if d == 0:
                score += 10
            else:
                score += 1.0/(d*d)
                
        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIdx, depth, isMax):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = gameState.getLegalActions(agentIdx)
        bestScore = 0
        bestAction = None
        # Pacman move
        if isMax:
            scores = [self.minimax(gameState.generateSuccessor(agentIdx, action), 1, depth - 1, False)[0] for action in actions]
            bestScore = max(scores)
            bestScoreIdxes = [i for i in range(len(scores)) if scores[i] == bestScore]
            bestAction = actions[bestScoreIdxes[0]]
        # ghost move
        else:
            scores = []
            # last ghost move
            if agentIdx == gameState.getNumAgents() - 1:
                scores = [self.minimax(gameState.generateSuccessor(agentIdx, action), 0, depth - 1, True)[0] for action in actions]
            else:
                scores = [self.minimax(gameState.generateSuccessor(agentIdx, action), agentIdx + 1, depth, False)[0] for action in actions]
            bestScore = min(scores)
            bestScoreIdxes = [i for i in range(len(scores)) if scores[i] == bestScore]
            bestAction = actions[bestScoreIdxes[0]]
        return bestScore, bestAction

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth * 2, True)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, state, agent, depth, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)
        isMax = agent == 0
        # pacman move
        if isMax:
            bestValue = -1e100
            bestActions = [] 
            for action in actions:
                value = self.alphabeta(state.generateSuccessor(agent, action), 1, depth - 1, alpha, beta)[0]
                alpha = max(alpha, value)
                if value > bestValue:
                    bestValue = value
                    bestActions = [action]
                elif value == bestValue:
                    bestActions.append(action)
                if beta < value:
                    break
            return bestValue, random.choice(bestActions)
        # ghost move
        else:
            bestValue = 1e100
            bestActions = []
            if agent == state.getNumAgents() - 1:
                for action in actions:
                    value = self.alphabeta(state.generateSuccessor(agent, action), 0, depth - 1, alpha, beta)[0]
                    beta = min(beta, value)
                    if value < bestValue:
                        bestValue = value
                        bestActions = [action]
                    elif value == bestValue:
                        bestActions.append(action)
                    if alpha > value:
                        break
            else:
                for action in actions:
                    value = self.alphabeta(state.generateSuccessor(agent,action), agent + 1, depth, alpha,beta)[0]
                    beta = min(beta, value)
                    if value < bestValue:
                        bestValue = value
                        bestActions = [action]
                    elif value == bestValue:
                        bestActions.append(action)
                    if alpha > value:
                        break
            return bestValue, random.choice(bestActions)

           
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, 0, self.depth * 2, -1e100, 1e100)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, state, agent, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        isMax = agent == 0
        actions = state.getLegalActions(agent)

        if isMax:
            scores = [self.expectimax(state.generateSuccessor(agent, action), 1, depth - 1)[0] for action in actions]
            bestScore = max(scores)
            bestScoreIdxes = [i for i in range(len(scores)) if scores[i] == bestScore]
            bestAction = actions[random.choice(bestScoreIdxes)]
            return bestScore, bestAction
        else:
            score = 0
            probability = 1.0 / len(actions)
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                val = 0
                if agent == state.getNumAgents() - 1:
                    val = self.expectimax(successor, 0, depth - 1)[0]
                else:
                    val = self.expectimax(successor, agent + 1, depth)[0]
                score += val * probability
            return score, Directions.STOP

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, self.depth * 2)[1]
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostState = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    
    for ghost in ghostState:
        d = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer != 0:
            score += 2200.0 / d
        else: 
            score -= 300

    for capsule in currentGameState.getCapsules():
        d = manhattanDistance(pacmanPos, capsule)
        if d <= 1:
            score += 100
        else:
            score += 10.0/d

    for food in food.asList():
        d = manhattanDistance(pacmanPos, food);
        if d == 0:
            score += 10
        else:
            score += 1.0/(d*d)
            
    return score

# Abbreviation
better = betterEvaluationFunction


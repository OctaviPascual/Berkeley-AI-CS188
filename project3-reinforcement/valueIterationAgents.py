# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def batchValueIterationStep(self):

        newValues = util.Counter()

        for state in self.mdp.getStates():

            if not self.mdp.isTerminal(state):
                newValues[state] = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    QValue = self.computeQValueFromValues(state, action)
                    newValues[state] = max(newValues[state], QValue)

        return newValues

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            self.values = self.batchValueIterationStep()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        values = util.Counter()
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            values[nextState] = prob * (reward + self.discount*self.values[nextState])
        return values.totalCount()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionValues = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            actionValues[action] = self.computeQValueFromValues(state, action)
        return actionValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def stateValueIterationStep(self, state):
        bestQValue = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            QValue = self.computeQValueFromValues(state, action)
            bestQValue = max(bestQValue, QValue)
        return bestQValue

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                self.values[state] = self.stateValueIterationStep(state)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computePredecessors(self):
        """
            Returns a dictionary where each key is a state and the value is a set
            containing all the predecessors of that state
        """

        # Store the predecessors of each state in a set to avoid duplicates
        predecessors = collections.defaultdict(set)

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    # Only add the states that have a nonzero probability of being reached
                    if prob > 0: predecessors[nextState].add(state)

        return predecessors

    def difference(self, state):
        """
            Returns the absolute value of the difference between the current value of state
            and the highest Q-Value across all possible actions from state
        """
        bestQValue = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            QValue = self.computeQValueFromValues(state, action)
            bestQValue = max(bestQValue, QValue)

        return abs(bestQValue - self.values[state])

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = self.computePredecessors()
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():

            if not self.mdp.isTerminal(state):
                diff = self.difference(state)
                # We use a negative diff because the priority queue is a min-heap
                # and we want to prioritize updating states that have a higher error,
                # thus a higher diff
                queue.push(state, -diff)

        for _ in range(self.iterations):

            if queue.isEmpty(): break

            state = queue.pop()

            # We can reuse the function that we used in AsynchronousValueIterationAgent
            self.values[state] = self.stateValueIterationStep(state)

            for predecessor in predecessors[state]:
                diff = self.difference(predecessor)
                # Pay attention on how the update function works
                if diff > self.theta: queue.update(predecessor, -diff)

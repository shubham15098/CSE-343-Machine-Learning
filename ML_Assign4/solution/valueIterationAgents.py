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
        
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        
        for i in range(0,self.iterations):
            max11 = -999999
            count = 0
            nn = util.Counter()
            # print nn
            c1 = []
            c1 = self.mdp.getStates()
            length1 = len(c1)
            
            while(1):
            	# print state
            	# print "yeah"
            	if(count >= length1):
            		break

            	state = c1[count]
            	count += 1

                actions = self.mdp.getPossibleActions(state)

                max11 = -999999
                # print self.mdp.getPossibleActions(state)

                for action in actions:
                    # print action
                    total = 0

                    list1 = []
                    list1 = self.mdp.getTransitionStatesAndProbs(state, action)
                    
                    length2 = len(list1)
                    count2 = 0 

                    while(1):
                    	if(count2 >= length2):
                    		break
                    	
                    	nxt , pp = list1[count2]
                    	count2 += 1

                    	total += pp * (self.mdp.getReward(state, action, nxt) + self.discount * self.values[nxt])

                    # v = self.computeQValueFromValues(state, action)

                    flag = 0

                    if(total > max11):
                    	flag = 1
                    else:
                    	flag = 0

                    
                    if flag == 1:
                        max11 = total
                        nn[state] = total
                        # print nn
                        
                	# print action
            self.values = nn

        # Write value iteration code here
        "*** YOUR CODE HERE ***"


    
    	


    


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
        total = 0
        list1 = []
        list1 = self.mdp.getTransitionStatesAndProbs(state, action)
        length2 = len(list1)
        count2 = 0 

        while(1):
        	if(count2 >= length2):
        		break

        	nxt , pp = list1[count2]
        	count2 += 1
        	total += pp * (self.mdp.getReward(state, action, nxt) + self.discount * self.values[nxt])


        return total
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # return self.actions[state]
        # print "yeah2"
        policy = 0
        x1 = self.mdp.isTerminal(state) 
        
        value = -999999
        actions = self.mdp.getPossibleActions(state)
        
        for action in actions:
          tmp = self.computeQValueFromValues(state, action)

          flag = 0

          if(tmp > value):
          	flag = 1
          else:
          	flag = 0

          if(flag == 1):
          	value = tmp
          	policy = action
          



          
        if(x1 == False):
        	return policy
        return 0
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

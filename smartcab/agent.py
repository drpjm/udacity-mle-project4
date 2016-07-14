import random
import math
import itertools as iters
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.trial_num = 0
        self.color = 'red'  # override color
        self.eps = 1 # some % of the time, choose a random action
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.possible_actions = [None, 'forward', 'left', 'right']
        # self.possbible_states = ['red', 'green']
        # Just an indicator of traffic: true or false.
        self.possbible_states = list(iters.product(['red','green'],[True,False]))
        # All three pathways.
        # self.possbible_states = list(iters.product(['red','green'],[True,False],[True,False],[True,False]))
        self.gamma = 0.5 # Discount factor
        self.curr_action = None
        self.qtable = {}
        # Init the qtable values to 0
        for s in self.possbible_states:
            for a in self.possible_actions:
                self.qtable[(s,a)] = 0.0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trial_num += 1
        # self.eps = 1.0 / math.sqrt(self.trial_num)
        self.eps -= 0.005
        print "*** TRIAL {}, eps={}".format(self.trial_num, self.eps)
        # print self.eps
        print self.qtable

    def update(self, t):
        # SENSE
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        alpha = 1.0 / (t + 1)
        # print alpha
        # State update:
        self.state = update_state_min(inputs)

        # print "*** Agent state = {}".format(self.state)

        # First: random action selection:
        # rand_action_idx = random.randint(0,3)
        # currAction = self.possible_actions[rand_action_idx]

        # ACT
        roll = random.random();
        if roll < self.eps:
            rand_action_idx = random.randint(0,3)
            # print "== Random move: {}".format(self.possible_actions[rand_action_idx])
            self.curr_action = self.possible_actions[rand_action_idx]
        else:
            qVals = np.zeros(len(self.possible_actions))
            for i in range(0, len(self.possible_actions)):
                qVals[i] = self.qtable[(self.state, self.possible_actions[i])]
            action_idx = np.argmax(qVals)
            # print "== Greedy move: {}".format(self.possible_actions[action_idx])
            self.curr_action = self.possible_actions[action_idx]

        # Execute action and get reward
        reward = self.env.act(self, self.curr_action)
        # print "*** Agent reward = {}".format(reward)

        # Update state *after* the action.
        new_state = update_state_min(self.env.sense(self))

        # Update the qtable accordingly
        sa_pair = (self.state, self.curr_action)
        # Q-table UPDATE
        currQ = self.qtable[sa_pair]
        # Compute max Q over all state, action pairs.
        qVals = np.zeros(len(self.possible_actions))
        for idx in range(0, len(self.possible_actions)):
            qVals[idx] = self.qtable[(new_state, self.possible_actions[idx])]
        maxNextQ = np.amax(qVals)
        self.qtable[sa_pair] = (1-alpha)*currQ + alpha*(reward + self.gamma*maxNextQ)

        # print self.qtable
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, self.curr_action, reward)  # [debug]

def update_state_min(inputs_dict):
        new_light = inputs_dict['light']
        has_traffic = False
        if inputs_dict['oncoming'] != None:
            has_traffic = True
        if inputs_dict['left'] != None:
            has_traffic = True
        if inputs_dict['right'] != None:
            has_traffic = True

        return (new_light, has_traffic)

def update_state(inputs_dict):
        new_light = inputs_dict['light']
        new_oncoming = False
        new_left = False
        new_right = False
        if inputs_dict['oncoming'] != None:
            new_oncoming = True
        if inputs_dict['left'] != None:
            new_left = True
        if inputs_dict['right'] != None:
            new_right = True

        return (new_light, new_oncoming, new_left, new_right)

def get_qtable(self):
    return self.qtable

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()

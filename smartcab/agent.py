import random
import csv
import math
import itertools as iters
import numpy as np
from smartcab.environment import Agent, Environment
from smartcab.planner import RoutePlanner
from smartcab.simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.trial_num = 0
        self.color = 'red'  # override color
        self.eps = 1 # some % of the time, choose a random action
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # self.net_reward = 0.0

        self.possible_actions = [None, 'forward', 'left', 'right']
        self.possible_waypoints = ['forward', 'left', 'right']

        # Only lights:
        # self.possbible_states = ['red', 'green']
        # Intersection traffic indicator: true or false.
        # self.possbible_states = list(iters.product(['red','green'],[True,False],self.possible_waypoints))
        # More detailed intersection traffic indicator:
        # self.possbible_states = list(iters.product(['red','green'],[True,False],[True,False],[True,False]))
        # Full detail: a LOT of states!
        # self.possbible_states = list(iters.product(['red','green'],self.possible_actions,self.possible_actions,self.possible_actions))
        # Adding waypoints to coarse intersection traffic model:
        self.possbible_states = list(iters.product(['red','green'],[True,False],[True,False],[True,False],self.possible_waypoints))

        print("Size of state space = {}".format(len(self.possbible_states)))

        self.gamma = 0.2 # Discount factor
        self.curr_action = None
        self.qtable = {}
        # Init the qtable values to 0
        for s in self.possbible_states:
            for a in self.possible_actions:
                self.qtable[(s,a)] = 0.0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trial_num += 1
        # Tweak to make exploration last a few more rounds.
        self.eps = 1.0 / math.sqrt(self.trial_num)
        self.alpha = 1.0 / self.trial_num
        print("*** TRIAL {}, eps={}".format(self.trial_num, self.eps))
        # print "*** Last net reward = {}".format(self.net_reward)
#        print_qtable(self.qtable)
        # Write out the net reward for the *last* trial.
        # f = open('reward.txt', 'a')
        # f.write(str(self.trial_num) + ',' + str(self.net_reward))
        # f.write('\n')
        # self.net_reward = 0.0

    def update(self, t):
        # SENSE
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # alpha = 1.0 / (t + 1)
        # print alpha
        # State update:
        # self.state = update_state_coarse(inputs, self.next_waypoint)
        self.state = update_state_fine(inputs, self.next_waypoint)
        # self.state = update_state_full(inputs)

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
            # print "== Greedy move: {} with value = {}".format(self.possible_actions[action_idx],
            # self.qtable[(self.state, self.possible_actions[action_idx])])
            self.curr_action = self.possible_actions[action_idx]

        # Execute action and get reward
        reward = self.env.act(self, self.curr_action)
        # self.net_reward += reward

        # Update state *after* the action.
        # new_state = update_state_coarse(self.env.sense(self), self.next_waypoint)
        new_state = update_state_fine(self.env.sense(self), self.next_waypoint)

        # Update the qtable accordingly
        sa_pair = (self.state, self.curr_action)
        # Q-table UPDATE
        Qcurr = self.qtable[sa_pair]
        # Compute max Q over all state, action pairs.
        QMax = -1000.0
        for idx in range(0, len(self.possible_actions)):
            curr = self.qtable[(new_state, self.possible_actions[idx])]
            if curr > QMax:
                QMax = curr

        self.qtable[sa_pair] = (1-self.alpha)*Qcurr + self.alpha*(reward + self.gamma*QMax)
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, self.curr_action, reward)  # [debug]

# If any direction has traffic, say it is True - too conservative.
def update_state_coarse(inputs_dict, waypoint):
        new_light = inputs_dict['light']
        has_traffic = False
        if inputs_dict['oncoming'] != None:
            has_traffic = True
        if inputs_dict['left'] != None:
            has_traffic = True
        if inputs_dict['right'] != None:
            has_traffic = True

        return (new_light, has_traffic ,waypoint)

# Abstracts traffic in each part of the intersection as "there" or "not there."
def update_state_fine(inputs_dict, waypoint):
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

        return (new_light, new_oncoming, new_left, new_right, waypoint)

# Full detail provided on traffic, not just a binary indicator.
def update_state_full(inputs_dict, waypoint):
        new_light = inputs_dict['light']
        new_oncoming = inputs_dict['oncoming']
        new_left = inputs_dict['left']
        new_right = inputs_dict['right']

        return (new_light, new_oncoming, new_left, new_right, waypoint)


def get_qtable(self):
    return self.qtable

def print_qtable(qtable):
    qtableStr = ""
    for k,v in qtable.iteritems():
        qtableStr += "{:<20} {:<20}\n".format(k, v)
    print(qtableStr)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.eps = 0.2 # some % of the time, choose a random action
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.possible_actions = [None, 'forward', 'left', 'right']
        self.possbible_states = ['red', 'green']
        self.qtable = {}
        # Init the qtable values to 0
        for s in self.possbible_states:
            for a in self.possible_actions:
                self.qtable[(s,a)] = 0.0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # State update:
        self.state = inputs['light'];

        # First: random action selection:
        rand_action_idx = random.randint(0,3)
        action = self.possible_actions[rand_action_idx]

        # Execute action and get reward
        reward = self.env.act(self, action)
        # Update the qtable accordingly
        sa_pair = (self.state, action)
        print "Qtable entry before: "
        print self.qtable[sa_pair]
        print "Qtable entry after: "
        self.qtable[sa_pair] = reward
        print self.qtable[sa_pair]

        # TODO: Learn policy based on state, action, reward
        roll = random.random();
        print roll
        if roll < self.eps:
            print "* Choose random action."
        else:
            print "* Q learning policy."

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()

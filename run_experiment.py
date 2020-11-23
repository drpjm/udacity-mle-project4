import argparse

from smartcab.agent import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Executes a simple Q-learning agent in the smart cab environment.")
    parser.add_argument('num_trials', type=int, default=10, help='Number of trials to train the agent.')
    parser.add_argument('sim_step', type=float, default=0.1, help='The simulation step size.')
    args = parser.parse_args()

    run(n_trials=args.num_trials, delta_t=args.sim_step)


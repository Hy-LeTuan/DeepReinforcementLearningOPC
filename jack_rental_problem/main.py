import argparse
import numpy as np
import os
from environment import Enviroment
from agent import Agent

parser = argparse.ArgumentParser()

parser.add_argument("--type", type=str, default="theoretical",
                    choices=["theoretical", "empirical"])
parser.add_argument("--constants", nargs='+', type=int,
                    help="List of contants for distributions in the problem, following this order:  rental lambda at location 1, rental lambda at location 2, return lambda at location 1, return lambda at location 2", default=[3, 4, 3, 2])
parser.add_argument("--policy_iterations", type=int, default=5,
                    help="max number of iterations for finding the optimal policy. Default = 5")
parser.add_argument("--cars_max", type=int, default=3,
                    help="Max number of cars at each location. Default = 3")

args = parser.parse_args()

if __name__ == "__main__":
    # initialize agent constants
    training_type = args.type
    max_number_of_cars = args.cars_max

    available_actions = np.stack([np.arange(
        0, max_number_of_cars + 1), np.arange(0, -max_number_of_cars - 1, step=-1)])

    # initialize environment constants
    expected_request_lambda_1, expected_request_lambda_2, expected_return_lambda_1, expected_return_lambda_2 = args.constants

    max_iterations = args.policy_iterations

    rewards = np.array([10, -2])

    # initialize agent
    agent = Agent(cars_max=max_number_of_cars, actions=available_actions, starting_policy=np.zeros(
        (max_number_of_cars + 1, max_number_of_cars + 1, 2), dtype=np.int32), states=np.zeros((max_number_of_cars + 1, max_number_of_cars + 1)), rewards=rewards, theta=0.001, gamma=0.9)

    # initialize environment
    environment = Enviroment(expected_request_lambda_1=expected_request_lambda_1, expected_request_lambda_2=expected_request_lambda_2,
                             expected_return_lambda_1=expected_return_lambda_1, expected_return_lambda_2=expected_return_lambda_2, agent=agent)

    # start training
    if training_type == "theoretical":
        iteration = environment.theoretical_train(
            max_iterations=max_iterations)
    else:
        iteration = environment.train(max_iterations=max_iterations)

    ####################################################

    if iteration != max_iterations:
        print(f"Took {iteration + 1} itertaions to find optmial policy")
    else:
        print(f"Max iterations reached for policy")

    policy_history = np.array(environment.policy_history)
    value_function = environment.agent.state_policy_values
    constants = np.array(
        [max_number_of_cars, expected_request_lambda_1, expected_request_lambda_2, expected_return_lambda_1, expected_return_lambda_2])

    if training_type == "theoretical":
        np.save("./theoretical/history.npy", policy_history)
        np.save("./theoretical/value_function.npy", value_function)
        np.save("./theoretical/actions.npy", available_actions)
        np.save("./theoretical/rewards.npy", rewards)
        np.save("./theoretical/constants.npy", constants)

    else:
        np.save("./empirical/history.npy", policy_history)
        np.save("./empirical/value_function.npy", value_function)
        np.save("./empirical/actions.npy", available_actions)
        np.save("./empirical/rewards.npy", rewards)
        np.save("./empirical/constants.npy", constants)

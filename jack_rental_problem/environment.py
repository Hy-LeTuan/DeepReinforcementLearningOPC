import numpy as np
from tqdm import tqdm
from agent import Agent


class Enviroment:
    def __init__(self, expected_request_lambda_1: float, expected_request_lambda_2: float, expected_return_lambda_1: float, expected_return_lambda_2: float, agent: Agent):
        # initialize Poisson random variables
        self.expected_request_lambda_1 = expected_request_lambda_1
        self.expected_request_lambda_2 = expected_request_lambda_2

        self.expected_return_lambda_1 = expected_return_lambda_1
        self.expected_return_lambda_2 = expected_return_lambda_2

        # intiailize agent
        self.agent = agent

        # initialize policy history
        self.policy_history = []

    def get_rental_requests(self) -> tuple:
        """
        -> returns a tuple of the number of rental requests at each location
        """
        rental_request_1 = np.random.poisson(
            lam=self.expected_request_lambda_1)
        rental_request_2 = np.random.poisson(
            lam=self.expected_request_lambda_2)

        return (rental_request_1, rental_request_2)

    def get_customer_returns(self) -> tuple:
        """
        -> returns a tuple of the number of cars returning at each location
        """
        customer_return_1 = np.random.poisson(
            lam=self.expected_return_lambda_1)
        customer_return_2 = np.random.poisson(
            lam=self.expected_return_lambda_2)

        return (customer_return_1, customer_return_2)

    def train(self, number_of_days, max_iterations) -> int:
        """
        -> returns the number of iteration needed to find an optimal policy, or the number of max iterations if multiple policies are found
        """
        for iteration in tqdm(range(max_iterations), desc="Policy Check", total=max_iterations):
            # save old policy and initialize benchmark variables
            benchmark_rental_request_1, benchmark_rental_request_2 = self.get_rental_requests()
            benchmark_customer_return_1, benchmark_customer_return_2 = self.get_customer_returns()

            self.policy_history.append(self.agent.policy)

            # inner loop to evaluate value function
            for d in tqdm(range(number_of_days), desc="Policy Evaluation", total=number_of_days):
                today_rental_request_1, today_rental_request_2 = self.get_rental_requests()
                today_customer_return_1, today_customer_return_2 = self.get_customer_returns()

                delta = self.agent.policy_evaluation_step(today_rental_request_1=today_rental_request_1, today_rental_request_2=today_rental_request_2,
                                                          today_customer_return_1=today_customer_return_1, today_customer_return_2=today_customer_return_2)

                if self.agent.check_valid_value_function_delta(delta=delta):
                    break

            # check if current policy is optimal policy
            has_optimal_policy = self.agent.policy_improvement_step(today_rental_request_1=benchmark_rental_request_1, today_rental_request_2=benchmark_rental_request_2,
                                                                    today_customer_return_1=benchmark_customer_return_1, today_customer_return_2=benchmark_customer_return_2)

            if has_optimal_policy:
                return iteration

        return max_iterations


if __name__ == "__main__":
    # initialize agent constants
    max_number_of_cars = 3
    available_actions = np.stack([np.arange(
        0, max_number_of_cars + 1), np.arange(0, -max_number_of_cars - 1, step=-1)])

    # initialize environment constants
    expected_request_lambda_1 = 3
    expected_request_lambda_2 = 4
    expected_return_lambda_1 = 3
    expected_return_lambda_2 = 2

    number_of_days = 100
    max_iterations = 5

    # initialize agent
    agent = Agent(cars_max=max_number_of_cars, actions=available_actions, starting_policy=np.zeros(
        (3, 3)), states=np.zeros((3, 3)), rewards=[0, 10, -2], theta=1e-3, gamma=0.9)

    # initialize environment
    environment = Enviroment(expected_request_lambda_1=expected_request_lambda_1, expected_request_lambda_2=expected_request_lambda_2,
                             expected_return_lambda_1=expected_return_lambda_1, expected_return_lambda_2=expected_return_lambda_2)

    # start training
    environment.train(number_of_days=number_of_days,
                      max_iterations=max_iterations)

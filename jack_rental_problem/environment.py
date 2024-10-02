import numpy as np
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

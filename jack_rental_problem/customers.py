import numpy as np


class Customers:
    def __init__(self, expected_request_lambda_1, expected_return_lambda_1, expected_request_lambda_2, expected_return_lambda_2):
        self.expected_request_lambda_1 = expected_request_lambda_1
        self.expected_request_lambda_2 = expected_request_lambda_2

        self.expected_return_lambda_1 = expected_return_lambda_1
        self.expected_return_lambda_2 = expected_return_lambda_2

    def get_rental_return(self):
        request1 = np.random.poisson(lam=self.expected_request_lambda_1)
        request2 = np.random.poisson(lam=self.expected_request_lambda_2)

        return1 = np.random.poisson(lam=self.expected_return_lambda_1)
        return2 = np.random.poisson(lam=self.expected_return_lambda_2)

        return (request1, return1), (request2, return2)

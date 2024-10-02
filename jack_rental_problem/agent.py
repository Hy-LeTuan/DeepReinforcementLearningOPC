import numpy as np


class Agent:
    def __init__(self, cars_max: int, actions: list, starting_policy: list, states: list, rewards: list, theta: float, gamma: float):
        # numbers of cars
        self.cars_max = cars_max

        # variables for reinforcement learing

        # policy contains index into the action array
        self.policy = np.array(starting_policy) if type(
            starting_policy) != np.ndarray else starting_policy

        # contains the actual value of increasing / decreasing the cars of the 2 locations
        self.actions = np.array(actions) if type(
            actions) != np.ndarray else actions

        # each state represents the number of cars in both locations
        self.states = np.array(states) if type(
            states) != np.ndarray else states

        # type of rewards available
        self.rewards = rewards

        self.theta = theta
        self.gamma = gamma

        self.state_policy_values = np.zeros_like(self.states)

        # the available states are the starting number of cars of each location to moving all cars from 1 place to another

    def get_reward(self, reward_index, number_of_cars: int):
        return self.rewards[reward_index] * number_of_cars

    def p(self, s, action_index: tuple, today_rental_request_1: int, today_rental_request_2: int, today_customer_return_1, today_customer_return_2) -> tuple:
        """
        calculate reward based on the number of rental request and the next state using number of customer returns
        -> returns a tuple of reward and next state
        """
        cars1 = s[0] + 1
        cars2 = s[1] + 1

        # calculate rewards based on today's number of cars
        number_of_cars_moved = self.actions[action_index]

        if cars1 < np.abs(number_of_cars_moved) or cars2 < np.abs(number_of_cars_moved):
            return None

        cars1 += number_of_cars_moved
        cars2 += number_of_cars_moved

        cost = self.get_reward(
            1, number_of_cars=np.abs(self.actions[action_index]))

        final_reward = np.float32(0)

        # if request > cars -> rent all cars. else, rent 'request' numbers of cars
        reward1 = self.get_reward(0, min(today_rental_request_1, cars1))
        # if request > cars -> rent all cars. else, rent 'request' numbers of cars
        reward2 = self.get_reward(0, min(today_rental_request_2, cars2))

        final_reward += reward1
        final_reward += reward2
        final_reward += cost

        # calculate next state using the number of cars returned
        cars1 += today_customer_return_1
        cars2 += today_customer_return_2

        cars1 = min(cars1, self.cars_max)
        cars2 = min(cars2, self.cars_max)

        return final_reward, (cars1 - 1, cars2 - 1)

    def get_action_from_policy(self, s: tuple) -> tuple:
        """
        -> return the index into the action array 
        """
        action_index = self.policy[s]
        action_index = tuple(action_index)

        return action_index

    def set_action_to_policy(self, action_index: tuple, s: tuple) -> None:
        """
        greedily set the new policy to the best action according to the value function 
        """
        self.policy[s] = action_index

    def check_valid_value_function_delta(self, delta) -> bool:
        if delta < self.theta:
            return True
        else:
            return False

    def policy_evaluation_step(self, today_rental_request_1, today_rental_request_2, today_customer_return_1, today_customer_return_2) -> np.float32:
        delta = np.float32(0)

        for i in range(self.state_policy_values.shape[0]):
            for j in range(self.state_policy_values.shape[1]):
                s = (i, j)
                v = self.state_policy_values[s]

                action_index = self.get_action_from_policy(s)

                # get reward and next state
                reward, next_state = self.p(s, action_index=action_index, today_rental_request_1=today_rental_request_1, today_rental_request_2=today_rental_request_2,
                                            today_customer_return_1=today_customer_return_1, today_customer_return_2=today_customer_return_2)

                # update current value function of current policy
                self.state_policy_values[s] = reward + \
                    self.gamma * self.state_policy_values[next_state]

                # calculate difference delta
                delta = max(delta, np.abs(v - self.state_policy_values[s]))

        return delta

    def policy_improvement_step(self, today_rental_request_1, today_rental_request_2, today_customer_return_1, today_customer_return_2) -> bool:
        """
        -> returns True if the best policy is found, returns false if the policy is not found
        """

        policy_stable = True

        for i in range(self.state_policy_values.shape[0]):
            for j in range(self.state_policy_values.shape[1]):
                s = (i, j)
                a = self.get_action_from_policy(s)

                # getting the best action for the current state based on the value function. argmax_a p(r, s' | s, a)
                best_action_index = (0, 0)
                best_action_value = np.float32(0)

                for x in range(self.actions.shape[0]):
                    for y in range(self.actions.shape[1]):
                        action_index = (x, y)

                        try:
                            reward, next_state = self.p(s=s, action_index=action_index, today_rental_request_1=today_rental_request_1, today_rental_request_2=today_rental_request_2,
                                                        today_customer_return_1=today_customer_return_1, today_customer_return_2=today_customer_return_2)
                        except TypeError:
                            continue

                        current_action_value = reward + \
                            self.gamma * self.state_policy_values[next_state]

                        if current_action_value > best_action_value:
                            best_action_value = current_action_value
                            best_action_index = action_index

                self.set_action_to_policy(best_action_index, s)

                print(
                    f"state: {s} with best_action_index: {best_action_index}")

                if a != best_action_index:
                    policy_stable = False

        return policy_stable

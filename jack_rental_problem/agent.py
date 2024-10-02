import numpy as np


class Agent:
    def __init__(self, cars1, cars2, cars_max: int, actions: list, starting_policy: list, states: list, rewards: list, theta: float, gamma: float):
        # numbers of cars
        self.cars1 = cars1
        self.cars2 = cars2
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

    def p(self, s, action_index: tuple, next_day_rental_request_1: int, next_day_rental_request_2: int) -> tuple:
        # do not include return because returns are only effective till the next day and is therefore added to the number of cars at each location at the end of the loop

        # the probability of getting to the next state is 1 -> return next state

        # what if the next day request is the actual request, and we assume that we actually know the environment and how it works ?

        # from the current state and selecting the action `action_index`, how does it go?
        # cars1 = self.cars1 + self.actions[action_index]
        # cars2 = self.cars2 - self.actions[action_index]

        cars1 = s[0]
        cars2 = s[1]

        number_of_cars_moved = self.actions[action_index]

        if cars1 < np.abs(number_of_cars_moved) or cars2 < np.abs(number_of_cars_moved):
            return None

        cars1 += number_of_cars_moved
        cars2 += number_of_cars_moved

        cost = self.get_reward(
            1, number_of_cars=np.abs(self.actions[action_index]))

        final_reward = np.float32(0)

        # if request > cars -> rent all cars. else, rent 'request' numbers of cars
        reward1 = self.get_reward(0, min(next_day_rental_request_1, cars1))
        # if request > cars -> rent all cars. else, rent 'request' numbers of cars
        reward2 = self.get_reward(0, min(next_day_rental_request_2, cars2))

        final_reward += reward1
        final_reward += reward2
        final_reward += cost

        return final_reward, (cars1 - 1, cars2 - 1)

    def get_action_from_policy(self, s: tuple) -> tuple:
        """
        -> return the index into the action array 
        """
        return self.policy[s]

    def policy_evaluation_step(self, next_day_rental_request_1, next_day_rental_request_2) -> np.float32:
        delta = np.float32(0)

        for i in range(self.state_policy_values.shape[0]):
            for j in range(self.state_policy_values.shape[1]):
                s = (i, j)
                v = self.state_policy_values[s]

                action_index = self.get_action_from_policy(s)

                # get reward and next state
                reward, next_state = self.p(
                    s, action_index=action_index, next_day_rental_request_1=next_day_rental_request_1, next_day_rental_request_2=next_day_rental_request_2)

                # update current value function of current policy
                self.state_policy_values[s] = reward + \
                    self.gamma * self.state_policy_values[next_state]

                # calculate difference delta
                delta = max(delta, np.abs(v - self.state_policy_values[s]))

        return delta

    def policy_evaluation(self, next_day_rental_request_1, next_day_rental_request_2) -> None:
        while True:
            delta = np.float32(0)

            # loop through each state
            # for i in range(self.state_policy_values.shape[0]):
            #     for j in range(self.state_policy_values.shape[1]):
            # s = (i, j)
            # v = self.state_policy_values[s]

            # # get action index from policy
            # action_index = self.get_action_from_policy(s)

            # # get reward and next state
            # reward, next_state = self.p(s, action_index=action_index, next_day_rental_request_1=next_day_rental_request_1, next_day_rental_request_2=next_day_rental_request_2)

            # # update current value function of current policy
            # self.state_policy_values[s] = reward + self.gamma * self.state_policy_values[next_state]

            delta = max(delta, self.policy_evaluation_step(
                next_day_rental_request_1, next_day_rental_request_2))

            if delta < self.theta:
                break

        return

    def policy_improvement_step(self, next_day_rental_request_1, next_day_rental_request_2) -> bool:
        """
        -> returns True if the best policy is found, returns false if the policy is not found
        """

        for i in range(self.state_policy_values.shape[0]):
            for j in range(self.state_policy_values.shape[1]):
                s = (i, j)
                a = self.get_action_from_policy(s)

                # getting the best action for the current state based on the value function
                best_action_index = np.zeros(2)
                best_action_value = np.float32(0)

                for x in range(self.actions.shape[0]):
                    for y in range(self.actions.shape[1]):
                        action_index = (x, y)
                        reward, next_state = self.p(
                            s=s, action_index=action_index, next_day_rental_request_1=next_day_rental_request_1, next_day_rental_request_2=next_day_rental_request_2)

                        current_action_value = reward + \
                            self.state_policy_values[next_state]

                        if current_action_value > best_action_value:
                            best_action_value = current_action_value
                            best_action_index = action_index

                if a != best_action_index:
                    return False
                else:
                    return True

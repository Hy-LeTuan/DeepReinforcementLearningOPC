import numpy as np


class BanditAgent:
    def __init__(self, name, actions: list, epsilon=None, UCB=None, UCB_constant=None):
        self.name = name
        self.actions = actions
        self.action_size = len(actions)

        self.action_value_table = np.zeros_like(actions, dtype=np.float32)
        self.action_count_table = np.zeros_like(actions, dtype=np.float32)

        self.greedy = []

        self.epsilon = epsilon
        self.UCB = UCB

        if self.UCB:
            self.UCB_value_table = np.zeros_like(actions)
            if UCB_constant:
                self.UCB_constant = UCB_constant
            else:
                self.UCB_constant = 1

    def get_UCB_value_constant(self, action_index, time_step):
        if self.action_count_table[action_index] == 0:
            return 1e7
        else:
            return self.UCB_constant + np.sqrt(np.log(time_step) / self.action_count_table[action_index])

    def update_action_value_table(self, action_index: int, reward, UCB=False) -> None:
        k = self.action_count_table[action_index] + 1

        # average reward estimate
        self.action_value_table[action_index] = self.action_value_table[action_index] + (
            (1 / k) * (reward - self.action_value_table[action_index]))

        # update action count
        self.update_action_count_table(action_index)

    def update_action_value_table_with_UCB(self, action_index: int, reward, time_step):
        """
        time_step = i + 1 for i in iterations starting with 0 
        """
        k = self.action_count_table[action_index] + 1

        # average reward estimate
        self.action_value_table[action_index] = self.action_value_table[action_index] + (
            (1 / k) * (reward - self.action_value_table[action_index]))

        # update action count
        self.update_action_count_table(action_index)

        # update UCB table

        for i, action in enumerate(self.actions):
            self.UCB_value_table[i] = self.action_value_table[i] + \
                self.get_UCB_value_constant(
                    action_index=i, time_step=time_step)

    def update_action_count_table(self, action_index) -> None:
        self.action_count_table[action_index] += 1

    def choose_action_from_value_table(self, UCB=False) -> int:
        if self.epsilon:
            choices = np.array([0, 1])
            choice = np.random.choice(choices, size=None, p=[
                                      1.0 - self.epsilon, self.epsilon])

            if choice == 0:
                action_index = np.argmax(self.action_value_table)
                self.greedy.append(1)
            else:
                action_index = np.random.randint(0, self.action_size)
                self.greedy.append(0)
        elif UCB:
            action_index = np.argmax(self.UCB_value_table)
        else:
            action_index = np.argmax(self.update_action_value_table)

        return action_index


if __name__ == "__main__":
    agent = BanditAgent(name="agent", actions=[
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9], epsilon=0.1)
    for i in range(20):
        a = agent.choose_action_from_value_table()
        print(a)

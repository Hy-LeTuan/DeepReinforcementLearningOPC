import numpy as np


class BanditAgent:
    def __init__(self, name, actions: list, epsilon=None):
        self.name = name
        self.actions = actions
        self.action_size = len(actions)

        self.action_value_table = np.zeros_like(actions, dtype=np.float32)
        self.action_count_table = np.zeros_like(actions, dtype=np.float32)

        self.greedy = []

        self.epsilon = epsilon

    def update_action_count_table(self, action_index) -> None:
        self.action_count_table[action_index] += 1

    def update_action_value_table(self, action_index: int, reward) -> None:
        k = self.action_count_table[action_index] + 1

        # average reward estimate
        self.action_value_table[action_index] = self.action_value_table[action_index] + (
            (1 / k) * (reward - self.action_value_table[action_index]))

        # update action count
        self.update_action_count_table(action_index)

    def choose_action_from_value_table(self) -> int:
        if self.epsilon:
            choices = np.array([0, 1])
            choice = np.random.choice(choices, size=None, p=[
                                      1.0 - self.epsilon, self.epsilon])

            if choice == 0:
                action_index = np.argmax(self.update_action_value_table)
                self.greedy.append(1)
            else:
                action_index = np.random.randint(0, self.action_size)
                self.greedy.append(0)

        else:
            action_index = np.armgax(self.update_action_value_table)

        return action_index


if __name__ == "__main__":
    agent = BanditAgent(name="agent", actions=[
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9], epsilon=0.1)
    for i in range(20):
        a = agent.choose_action_from_value_table()
        print(a)

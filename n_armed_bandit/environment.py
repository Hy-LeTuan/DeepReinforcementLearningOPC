import time
import numpy as np
from tqdm.auto import tqdm
from agent import BanditAgent
from n_arm_bandit import NArmedBandit


class BanditEnvironment:
    def __init__(self, agent: BanditAgent, bandit: NArmedBandit):
        self.agent = agent
        self.bandit = bandit

    def train(self, steps: int = 0) -> list:
        history = []
        actions = []

        for step in range(steps):
            # choose the lever to pull
            action = self.agent.choose_action_from_value_table()

            # bandit gives out result for pulling a specific lever
            result, indices = self.bandit.pull_lever(
                index=action, return_indices=True)

            # display the symbols
            self.bandit.formatted_display(result=result, index=step, offset=0)

            # bandit calculate reward
            reward = self.bandit.calculate_reward(result, lever_index=action)

            # update action estimation based on new reward
            self.agent.update_action_value_table(
                action_index=action, reward=reward)

            history.append(reward)
            actions.append(action)

        print(end="\n")
        print(end="\n")
        print(end="\n")

        return history, actions

    def train_with_UCB(self, steps: int = 0) -> list:
        history = []
        actions = []

        for step in range(steps):
            # choose the lever to pull
            action = self.agent.choose_action_from_value_table(UCB=True)

            # bandit gives out result for pulling a specific lever
            result, indices = self.bandit.pull_lever(
                index=action, return_indices=True)

            # display the symbols
            self.bandit.formatted_display(result=result, index=step, offset=0)

            # bandit calculate reward
            reward = self.bandit.calculate_reward(result, lever_index=action)

            # update action estimation based on new reward
            self.agent.update_action_value_table_with_UCB(
                action_index=action, reward=reward, time_step=step + 1)

            history.append(reward)
            actions.append(action)

        print(end="\n")
        print(end="\n")
        print(end="\n")

        return history, actions


if __name__ == "__main__":
    agent = BanditAgent(name="agent", actions=[
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9], epsilon=None, UCB=True, UCB_constant=2)
    bandit = NArmedBandit(arms=10, reel_numbers=3)

    environment = BanditEnvironment(agent=agent, bandit=bandit)

    history, action = environment.train_with_UCB(steps=2000)
    history = np.array(history)
    action = np.array(action)

    np.save("./history.npy", history)
    np.save("./actions.npy", action)

    value_table = agent.action_value_table
    mean_deviations = np.array(bandit.mean_deviation_pairs)
    rewards = bandit.rewards
    greedy = agent.greedy

    np.save("./value_table.npy", value_table)
    np.save("./mean_deviation_pairs.npy", mean_deviations)
    np.save("./rewards.npy", rewards)
    np.save("./greedy", greedy)

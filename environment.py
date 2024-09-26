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

            time.sleep(0.1)

        print(end="\n")
        print(end="\n")
        print(end="\n")

        return history


if __name__ == "__main__":
    agent = BanditAgent(name="agent", actions=[
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9], epsilon=0.1)
    bandit = NArmedBandit(arms=10, reel_numbers=3)

    environment = BanditEnvironment(agent=agent, bandit=bandit)

    history = environment.train(steps=1000)
    history = np.array(history)

    # save history
    np.save("./history_2.npy", history)

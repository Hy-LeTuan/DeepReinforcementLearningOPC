import numpy as np
import time


class NArmedBandit:
    def __init__(self, arms: int = 10, reel_numbers: int = 3, symbols=None, rewards=None):
        """
        arms: number of levers
        reel_numbers: number of slots on the machine
        """

        self.reel_numbers = reel_numbers

        if not symbols:
            self.symbols = np.array(
                ["~", "!", "@", "#", "$", "%", "^", "&", "*", "S"])
            self.symbol_count = 10
        else:
            self.symbols = np.array(symbols) if type(
                symbols) == list else symbols
            self.symbol_count = len(symbols)

        if not rewards:
            self.rewards = np.random.rand(10)
        else:
            self.rewards = rewards

        self.arms = arms
        self.mean_deviation_pairs = []
        self.generate_arms(self.arms)

    def generate_arms(self, arms) -> None:
        # store mean and std deviation pair for distribution of different arms
        for i in range(arms):
            mean = np.random.randint(0, self.symbol_count)
            deviation = np.random.random()

            self.mean_deviation_pairs.append((mean, deviation))

    def pull_lever(self, index: int, return_indices=False) -> np.array:
        if index < 0 or index >= self.symbol_count:
            raise IndexError("Index exceeds the amount of lever available")

        # get mean and deviation specific for each levers
        mean, deviation = self.mean_deviation_pairs[index]

        # convert random numbers to appropriate indices
        random_numbers = np.random.normal(
            loc=mean, scale=deviation, size=self.reel_numbers)

        random_indices = [np.abs(int(np.floor(num % 10)))
                          for num in random_numbers]
        return_symbols = self.symbols[random_indices]

        # unallocate memory for efficient computing
        del random_numbers
        del mean
        del deviation

        if return_indices:
            return [return_symbols, random_indices]
        else:
            del random_indices
            return return_symbols

    def calculate_reward(self, result, lever_index):
        reward = 0.0
        existing_symbols = {}

        for i, symbol in enumerate(result):
            index = np.where(self.symbols == symbol)[0][0]
            reward += self.rewards[index]

        return reward

    def display_result(self, result: np.array, only_content=False) -> None:
        machine_width = 2 * len(result) - 1

        # calculate display row
        display_row = "|"

        for i in range(len(result)):
            if i == 0:
                display_row += f" {result[i]}"
            else:
                display_row += f" | {result[i]}"
        display_row += " |"

        if only_content:
            print(display_row, end="\r")
        else:
            print(f"| {'- ' * machine_width}|")
            print(f"| {'- ' * machine_width}|")
            print(display_row)
            print(f"| {'- ' * machine_width}|")
            print(f"| {'- ' * machine_width}|")

    def formatted_display(self, result, index, offset=0):
        """
        result: symbols after 1 time of rolling
        index: step of the training loop
        offset: number of print statements in your training loop
        """
        LINE_CLEAR = '\x1b[2K'
        LINE_UP = '\033[1A'
        if index == 0:
            self.display_result(result, only_content=False)
            for i in range(2 + offset):
                print(LINE_UP, end="")

            print(LINE_UP, end=LINE_CLEAR)
        else:
            self.display_result(result, only_content=True)

    def play(self, iterations: int, scoreboard: np.array = None, delay: int = 0.015) -> np.array:
        LINE_CLEAR = '\x1b[2K'
        LINE_UP = '\033[1A'

        if scoreboard == None:
            scoreboard = np.zeros((self.arms, self.symbol_count))

        for i in range(self.arms):
            if i != 0:
                print(end="\n")
                print(end="\n")
                print(end="\n")

            print(f"Pulling lever number {i}")
            for j in range(iterations):
                result, indices = self.pull_lever(index=i, return_indices=True)

                # update scoreboard
                scoreboard[i, indices] += 1

                if j == 0:
                    self.display_result(result, only_content=False)
                    print(LINE_UP, end="")
                    print(LINE_UP, end="")
                    print(LINE_UP, end=LINE_CLEAR)
                else:
                    self.display_result(result, only_content=True)

                time.sleep(delay)

        np.save("./scoreboard.npy", scoreboard)
        return scoreboard


if __name__ == "__main__":
    bandit = NArmedBandit(arms=10, reel_numbers=3)
    # scoreboard = bandit.play(iterations=1000, delay=0.0000)
    result, indices = bandit.pull_lever(index=0, return_indices=True)
    bandit.display_result(result)

    reward = bandit.calculate_reward(result, lever_index=0)

    print(reward)

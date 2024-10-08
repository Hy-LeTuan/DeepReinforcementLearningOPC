# Jack's Rental Problem

## Problem description

Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and is credited $\$10$ by the national company. If he is out of cars at that location, then the business is lost. Cars become available for renting the day after they are returned.

To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of $\$2$ per car moved. We assume that the number of cars requested and returned at each location are **Poisson random variables**, meaning that the probability that the number is $n$ is $\frac{\lambda^n}{n!}e^{-\lambda}$ , where $\lambda$ is the expected number.

Suppose $\lambda$ is $3$ and $4$ for rental requests at the first and second locations and $3$ and $2$ for returns. To simplify the problem slightly, we assume that there can be no more than $20$ cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of $5$ cars can be moved from one location to the other in one night. We take the discount rate to be $\gamma = 09$ and formulate this as a continuing finite **MDP**, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight.

## Approach

### Initialization

The number of states and actions in this problem is finite and is relatively small. This enables us to represent both state and action as $2\text{D}$ matrices (tabular method).

The states are represented as a matrix $S \in \mathbb{R}^{20 \times 20}$, with each row and column representing the number of cars at both location 1 and 2.

The actions are represented as a matrix $A \in \mathbb{R}^{6, 6}$ in which the first row stores numbers of cars moved from location 1 to location 2 and the second row stores number of cars moved from location 2 to location 1.

The policy can also be represented as a matrix with the same shape as state matrix $S$, which each cell pointing to the action that the agent should take when encountering a specific state $s$.

### Probability analysis

In this problem, we can cleary see that the number of rental requests and number of returns between 2 locations are different, ultimately leading to a different number of rewards after a certain amount of days.

Here, the reward is calculated based on the number of requests and returns of both locations, thus is under the influence of 4 different probabilities.

### Policy Evaluation

In policy evaluation, we evaluate the policy until the value funcitons accurately represent the expected reward. To accomplish this, we continuously backup the value function until the difference between new predictions and old predictions is less than a small $\Delta$.

Mathematically, value fucntion update in this problem is represented as

$$
v_\pi(s) = \sum_{\text{req}_1}p(\text{req}_{1}) \sum_{\text{req}_2}p(\text{req}_2) \sum_{\text{ret}_1}p(\text{ret}_1) \sum_{\text{ret}_2}p(\text{ret}_2) \left[r(\text{req}_1 \text{req}_2, \text{ret}_1, \text{ret}_2 \vert a) + \gamma v_\pi(\text{ret}_1 + \text{ret}_2 + a) \right]
$$

with $\text{req}_1$, $\text{req}_2$ being the number of rental requets, $\text{ret}_1$, $\text{ret}_2$ being the number of returns and $a$ being the action chosen by the policy (the number of cars moved).

### Policy Iteration

After having an accurate value function for the current policy, we greedily select the best action in each state according to the value function (which can be a totally different action if based on the policy) and change the policy of the current state to the best action, thus resulting in a new and strictly improved policy.

## Instructions

To run the simulation, navigate to the `jack_rental_problem` directory:

```
cd _jack_rental_problem
```

Then, execute the file `main.py` with the appropriate arguments to your experiment. In this setup, I explicitly remove the 5 cars moved maximum cap of the action to let you freely decide.

```
python ./main.py --type theoretical --constants 3, 4, 3, 2 --policy_iteration 5 --cars_max 3
```

The `type` flag chooses between calculating the expected reward using the theoretical formula above, or to calculate it as a simulatino (less accurate).

The `constants` flag contains the $\lambda$ variable for the Poisson distribution. The flag follows the order of [rental return 1, rental return 2, rental request 1, rental request 2].

The `policy_iteration` flag sets the maximum number of policy iteration to find the most optimal policy.

The `cars_max` flag sets the maxmimum number of car at each location.

## Visualization

To visualize your training result, navigate to the `result_visualization.ipynb` file and execute the cells there.

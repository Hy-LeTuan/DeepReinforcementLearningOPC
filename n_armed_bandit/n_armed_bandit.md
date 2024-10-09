# N-armed Bandit implementation

The N-armed Bandit implementation attempts to answer the exploration exploitation dilemma in Reinforcement Learning.

A RL agent needs to trade off between exploring uncertain policies and exploiting the current best policy. This trade off is a classic dilemma in RL. A simple approach to this problem is $\epsilon-\text{greedy}$, where $\epsilon \in (0, 1)$ usually a small value close to 0. In $\epsilon-\text{greedy}$, the agent selects a greedy action

$$
a = \text{argmax}_{a \in A}Q(s, a)
$$

for state $s$ with probability $1 - \epsilon$ and select a random action with probability $\epsilon$.

You can see the results of this $\epsilon$-greedy method by running the program and setting any arbitrary epsilon. With higher epsilons, you should see a larger fluctuations of the result due to the agent trying many other actions that are much less optimal and yield a much lower result.

In this simple style of random exploration, the non-greedy action are indiscriminately chosen from a distribution and each has a chance of $1 - \epsilon$ of being selected. However, it would be better to select the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates. One effective way of doing this is to select action as:

$$
A_t = \text{argmax}_a\left[Q_t(a) + \underbrace{c \sqrt{\frac{\ln t}{N_t(a)}}}_{\text{uncertainty term}}\right]
$$

In this equation, actions that are explored fewer times will have a high uncertainty term, causing the model to notice these actions more. However, because the value is combined with the actual estimates, actions where the estimated values are small will be discarded regardless of their uncertainties because they are deemed inefficient.

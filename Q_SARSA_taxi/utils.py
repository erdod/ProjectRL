import matplotlib.pyplot as plt

def plot_results(q_rewards, q_lengths, sarsa_rewards, sarsa_lengths, mc_rewards, mc_lengths, episodes):
    """
    Plots the results of the reinforcement learning algorithms.
    """
    lst = list(range(1, episodes + 1))

    plt.figure()
    plt.plot(lst[:episodes], q_rewards[:episodes])
    plt.plot(lst[:episodes], sarsa_rewards[:episodes])
    plt.plot(lst[:episodes], mc_rewards[:episodes])
    plt.legend(["Q-Learning", "SARSA", "MC Every Visit"])
    plt.title("Reinforcement Learning on Taxi Environment")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward for an episode")

    plt.figure()
    plt.title("Monte-Carlo on Taxi Environment")
    plt.plot(lst[:episodes], mc_rewards[:episodes])
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward for an episode")

    plt.figure()
    plt.title("Q-Learning on Taxi Environment")
    plt.plot(lst[:episodes], q_rewards[:episodes])
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward for an episode")

    plt.figure()
    plt.title("SARSA on Taxi Environment")
    plt.plot(lst[:episodes], sarsa_rewards[:episodes])
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward for an episode")

    plt.figure()
    plt.plot(lst[:episodes], q_lengths[:episodes])
    plt.plot(lst[:episodes], sarsa_lengths[:episodes])
    plt.plot(lst[:episodes], mc_lengths[:episodes])
    plt.legend(["Q-Learning", "SARSA", "MC Every Visit"])
    plt.title("Reinforcement Learning on Taxi Environment")
    plt.xlabel("Episode Number")
    plt.ylabel("Number of Steps (epoch)")

    plt.figure()
    plt.title("Monte-Carlo on Taxi Environment")
    plt.plot(lst[:episodes], mc_lengths[:episodes])
    plt.xlabel("Episode Number")
    plt.ylabel("Number of Steps (epoch)")

    plt.figure()
    plt.title("Q-Learning on Taxi Environment")
    plt.plot(lst[:episodes], q_lengths[:episodes])
    plt.xlabel("Episode Number")
    plt.ylabel("Number of Steps (epoch)")

    plt.figure()
    plt.title("SARSA on Taxi Environment")
    plt.plot(lst[:episodes], sarsa_lengths[:episodes])
    plt.xlabel("Episode Number")
    plt.ylabel("Number of Steps (epoch)")

    plt.show()

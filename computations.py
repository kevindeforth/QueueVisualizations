from mpmath import mp

mp.dps = 80


class ProbNCustomerTable:
    """
    This class computes the probabilities of having n customers in a M/M/1/K Queue system
    """

    def __init__(self, queue_capacity, rho):
        self.rho = rho
        self.p_table = [
            (mp.mpf(1) - self.rho) / (1 - self.rho**(queue_capacity + 1))
        ]

    def get(self, n):
        """
        returns the probability that n customers will be in the system at the same time
        """
        current_len = len(self.p_table)
        for i in range(current_len, n + 1):
            self.p_table.append(self.p_table[i - 1] * self.rho)

        return self.p_table[n]


class PartialExpTaylorSumTable:
    """
    This is a helper class and computes a partial sum of the Taylor expansion of the exponential function:
        Sum_k=0^n x^k/k!
    """

    class FactorialTable:
        """
        This is a helper class, computing x^k/k!
        """

        def __init__(self, x):
            self.x = x
            self.factorial_table = [mp.mpf(1)]

        def get(self, k):
            """
            returns x^k/k!
            """
            current_len = len(self.factorial_table)
            for i in range(current_len, k + 1):
                self.factorial_table.append(self.factorial_table[i - 1] *
                                            (self.x / mp.mpf(i)))

            return self.factorial_table[k]

    def __init__(self, x):
        self.factorial_table = self.FactorialTable(x)
        self.partial_sum = [self.factorial_table.get(0)]

    def get(self, n):
        """
        returns sum_k=0^n x^k/k!
        """
        current_len = len(self.partial_sum)
        for k in range(current_len, n + 1):
            self.partial_sum.append(self.partial_sum[k - 1] +
                                    self.factorial_table.get(k))
        return self.partial_sum[n]


class MM1KQueues:
    """
    This class simulates a M/M/1/K queues, where:
        - Arrivals follow an exponential distribution of rate `λ`;
        - Servicing follows an exponential distribution of rate `μ`;
        - There is a single server;
        - The queue is finite, accomodating a maximum of K customers (K-1 in the queue, 1 getting serviced).
    c.f. https://real-statistics.com/probability-functions/queueing-theory/m-m-1-k-queueing-model/#:~:text=The%20M%2FM%2F1%2F,K%20customers%20in%20the%20system.
    """

    def p_w(self, k, t, μ, λ):
        """
        computes the probability that the waiting time in the system exceeds `t`
        """
        max_queue_capacity = k
        rho = λ / μ
        if (max_queue_capacity, rho) not in self.p_tables:
            self.p_tables[(max_queue_capacity,
                           rho)] = ProbNCustomerTable(max_queue_capacity, rho)
        p_table = self.p_tables[(max_queue_capacity, rho)]
        x = μ * t
        if x not in self.taylor_sum_tables:
            self.taylor_sum_tables[x] = PartialExpTaylorSumTable(x)

        taylor_sum_table = self.taylor_sum_tables[x]

        coeff = mp.exp(-(μ * t)) / (1 - p_table.get(max_queue_capacity))
        res = mp.mpf(0)
        for i in range(0, max_queue_capacity):
            res += p_table.get(i) * taylor_sum_table.get(i)
        p = coeff * res
        #print(f"k: {max_queue_capacity}, λ: {λ}, μ: {μ}, t: {t}, p_w(t): {p}")
        return p

    def __init__(self):
        self.p_tables = {}
        self.taylor_sum_tables = {}
        #self.p_table = ProbNCustomerTable(k, λ, μ)
        #self.taylor_sum_table = PartialExpTaylorSumTable(μ * t, k)


import pickle


def save_class(queue):
    with open("queue_state.pkl", "wb") as f:
        pickle.dump(queue, f)


def load_class():
    try:
        with open("queue_state.pkl", "rb") as f:
            loaded_queue = pickle.load(f)
            return loaded_queue
    except:
        return MM1KQueues()


queues = load_class()  #MM1KQueues()

timeout = mp.mpf(200)
k_values = [5000, 10000, 20000, 30000, 40000, 80000]
mu_values = [50, 100, 150, 200]
lambda_values = [(i * 10 + 1) for i in range(21)]
probabilities = {μ: {k: [] for k in k_values} for μ in mu_values}
for μ in mu_values:
    for k in k_values:
        for λ in lambda_values:
            prob = queues.p_w(k=k, λ=mp.mpf(λ), μ=mp.mpf(μ), t=timeout)
            probabilities[μ][k].append(prob)
save_class(queues)
import matplotlib.pyplot as plt

rows, cols = 2, 2
assert rows * cols == len(mu_values)
fig, axes = plt.subplots(rows, cols,
                         figsize=(12, 12))  # One row, multiple columns

#from scipy.interpolate import interp1d

line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
markers = ['o', 's', 'D', '^', 'v']
#markers = ['']
axes = axes.flatten()
#import numpy as np

#lambda_smooth = np.linspace(min(lambda_values), max(lambda_values),
#                            300)  # 300 points for smoothness
for i, μ in enumerate(mu_values):  # Assign each plot to an axis
    ax = axes[i]
    for j, k in enumerate(k_values):
        #f_interp = interp1d(lambda_values, probabilities[μ][k], kind='cubic')
        #smooth_prob = f_interp(lambda_smooth)
        ax.plot(lambda_values,
                probabilities[μ][k],
                marker=markers[j % len(markers)],
                linestyle=line_styles[j % len(line_styles)],
                alpha=0.7,
                label=f'K={k}')
        #ax.plot(lambda_smooth,
        #        smooth_prob,
        #        marker=markers[j % len(markers)],
        #        linestyle=line_styles[j % len(line_styles)],
        #        alpha=0.5,
        #        label=f'K={k}')

    # Formatting for each subplot
    ax.set_xlabel("λ (Arrival Rate)")
    ax.set_ylabel("Timeout Probability")
    ax.set_title(f"μ = {μ}")
    ax.legend()
    ax.grid(True)
    #ax.set_yscale("log")

fig.suptitle("Timeout Probability for serving rate μ by arrival rate λ",
             fontsize=16,
             fontweight="bold")
# Adjust layout
plt.tight_layout()
plt.savefig("all_plots.png")  # Save all plots in a single image
# Second plot: One subplot per K value, different curves for μ values
rows, cols = 2, 3
assert rows * cols == len(k_values)
fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
axes = axes.flatten()

for i, k in enumerate(k_values):
    ax = axes[i]
    for j, μ in enumerate(mu_values):
        #f_interp = interp1d(lambda_values, probabilities[μ][k], kind='cubic')
        #smooth_prob = f_interp(lambda_smooth)
        ax.plot(lambda_values,
                probabilities[μ][k],
                marker=markers[j % len(markers)],
                linestyle=line_styles[j % len(line_styles)],
                alpha=0.7,
                label=f'μ={μ}')
        #ax.plot(lambda_smooth,
        #        smooth_prob,
        #        marker=markers[j % len(markers)],
        #        linestyle=line_styles[j % len(line_styles)],
        #        alpha=0.5,
        #        label=f'μ={μ}')

    ax.set_xlabel("λ (Arrival Rate)")
    ax.set_ylabel("Timeout Probability")
    ax.set_title(f"K = {k}")
    ax.legend()
    ax.grid(True)

fig.suptitle("Timeout Probability for queue size K by arrival rate λ",
             fontsize=16,
             fontweight="bold")
plt.tight_layout()
plt.savefig("all_plots_K.png")
#plt.show()
#for μ in mu_values:
#    plt.figure(figsize=(8, 6))  # Create a new figure for each μ
#
#    for k in k_values:
#        plt.plot(lambda_values,
#                 probabilities[μ][k],
#                 marker='o',
#                 linestyle='-',
#                 label=f'K={k}')
#
#    # Formatting
#    plt.xlabel("λ (Arrival Rate)")
#    plt.ylabel("Probability")
#    plt.title(f"Queue Waiting Probability (μ={μ})")
#    plt.legend()
#    plt.grid(True)
#
#    # Show the plot
#    plt.savefig("plot.png")
#    plt.show()
#queues.p_w(k=5000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=10000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=20000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=30000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=40000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=50000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=55000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=56000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=57000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=58000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=59000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=60000, λ=mp.mpf(601), μ=mp.mpf(300), t=mp.mpf(200))
#queues.p_w(k=5000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=10000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=20000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=30000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=40000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=50000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=55000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=56000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=57000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=58000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=59000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))
#queues.p_w(k=60000, λ=mp.mpf(601), μ=mp.mpf(100), t=mp.mpf(200))

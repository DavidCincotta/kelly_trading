# martingale strategy simulator
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import numpy as np
import math

def fn_apply(x, y, f):
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = f(x[i], y[j])
    return z

def martingale(bankroll, init_wager_percent, win_prob, num_wagers):
    wager = init_wager_percent * bankroll
    for _ in range(num_wagers):
        if random.random() < win_prob:
            bankroll += wager
            wager = init_wager_percent * bankroll
        else:
            bankroll -= wager
            wager = 2 * wager
        if wager > bankroll:
            wager = bankroll
    if bankroll <= 0:
        return 0
    return bankroll

def proportion_bet(bankroll, num_wagers, **kwargs):
    proportion = kwargs["proportion"]
    win_prob = kwargs["win_prob"]
    if win_prob < 0.5:
        return bankroll
    wager = bankroll * proportion
    for _ in range(num_wagers):
        if random.random() < kwargs.get("perterbation",lambda x: x)(win_prob):
            bankroll += wager
        else:
            bankroll -= wager
        wager = bankroll * proportion
        if wager > bankroll:
            wager = bankroll
    if bankroll <= 0:
        return 0
    return bankroll

def fixed_bet(bankroll, num_wagers, **kwargs):
    init_wager_percent = kwargs["proportion"]
    win_prob = kwargs["win_prob"]
    if win_prob < 0.5:
        return bankroll
    wager = init_wager_percent * bankroll
    for _ in range(num_wagers):
        if random.random() < kwargs.get("perterbation",lambda x: x)(win_prob):
            bankroll += wager
        else:
            bankroll -= wager
        if wager > bankroll:
            wager = bankroll
    if bankroll <= 0:
        return 0
    return bankroll

def kelly_bet(bankroll, num_wagers, **kwargs):
    win_prob = kwargs["win_prob"]
    odds = kwargs.get("odds", 2)
    c_kelly = kwargs["c_kelly"]
    # print(odds, win_prob, kelly_frac)
    for _ in range(num_wagers):

        win_prob_guess = kwargs.get("uncertainty",lambda x: x)(win_prob)
        kelly_frac = (win_prob_guess * odds - (1 - win_prob_guess)) / odds
        kelly_frac = np.clip(kelly_frac, 0, 1)
        wager = bankroll * kelly_frac * c_kelly
        assert wager <= bankroll
        if random.random() < kwargs.get("perterbation",lambda x: x)(win_prob):
            bankroll += wager*(odds-1)
        else:
            bankroll -= wager*(odds-1)
    if bankroll <= 0:
        return 0
    return bankroll


# Run the simulation
num_turns = 10
max_iter = 10000
N = int(max_iter/num_turns) # This sets a dynamic number of samples to keep a constant runtime
dims = 20
starting_bankroll = 1
bank_multiplier_global = 1.1
uncertainty=lambda x: random.uniform(-0.5,0.5)+x
perterbation=lambda x: random.uniform(-0.1,0.1)+x

plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(0.5, 1.0, dims) # win probability
# y = np.linspace(0.001, 1.0, dims) # proportion
y = np.linspace(1, 100, dims,dtype=int) # proportion
X, Y = np.meshgrid(x, y)
result_types = {0:"Mean B", 1:"Var B", 2:"Median B", 3:"Lowest B",4:"Highest B", 5:"P(Final B < 0.01S)", 6:"P(B >= S)", 7:"P(B < S)", 8:f"P(B >= KS)", 9:f"P(B < KS)"}
result_type = 0
ax.set_xlabel("Base Win Probability")
ax.set_ylabel("\n\n\nProportion:\nKelly Fraction\nFixed Fraction of inital bankroll\nFixed Proportion of bankroll") 
cur_title = ""

def bet_sim(betting_strategy, **kwargs):
    values = []
    local_N = max_iter/kwargs.get("num_turns",num_turns)
    local_N = int(local_N)
    local_N = max(local_N, 1)
    local_N = kwargs.get("N",local_N)
    for _ in range(local_N):
        temp = betting_strategy(starting_bankroll, kwargs.get("num_turns",num_turns), **kwargs)
        values.append(temp)
    result_type = kwargs.get("result_type",0)
    if result_type == 0:
        return np.mean(values)
    if result_type == 1:
        return np.var(values)
    if result_type == 2:
        return np.median(values)
    if result_type == 3:
        return np.min(values)
    if result_type == 4:
        return np.max(values)
    if result_type == 5:
        return len(list(filter(lambda a: a <= 0.01, values)))/local_N
    if result_type == 6:
        return len(list(filter(lambda a: a >= starting_bankroll, values)))/local_N
    if result_type == 7:
        return len(list(filter(lambda a: a < starting_bankroll, values)))/local_N
    if result_type == 8:
        return len(list(filter(lambda a: a >= starting_bankroll*kwargs.get("bank_multiplier",bank_multiplier_global), values)))/local_N
    if result_type == 9:
        return len(list(filter(lambda a: a < starting_bankroll*kwargs.get("bank_multiplier",bank_multiplier_global), values)))/local_N
    return -1


from typing import Callable
def plot_func(betting_strategy:Callable, **kwargs):
    Z= np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            Z[i][j] = bet_sim(betting_strategy,  proportion=y[i], win_prob=x[j], c_kelly=y[i], **kwargs)
    local_title = betting_strategy.__name__ +" "+result_types[kwargs.get("result_type",0)].replace("K",f'{kwargs.get("bank_multiplier",bank_multiplier_global)}')
    local_title+="\n"
    ax.plot_wireframe(X, Y, Z, color=kwargs.get("color","blue"),label=local_title,alpha=kwargs.get("alpha", 0.5))

def plot_diff(betting_strategy1:Callable,betting_strategy2:Callable, strat1_args:dict={}, strat2_args:dict={}, **kwargs):
    Z= np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            Z[i][j] = bet_sim(betting_strategy1,  proportion=y[i], win_prob=x[j], c_kelly=y[i], **strat1_args) - \
                    bet_sim(betting_strategy2,  proportion=y[i], win_prob=x[j], c_kelly=y[i], **strat2_args)
    local_title = betting_strategy1.__name__ +" "+result_types[strat1_args.get("result_type",result_type)].replace("K",f'{strat1_args.get("bank_multiplier",bank_multiplier_global)}') + " - " +betting_strategy2.__name__ +" "+ result_types[strat2_args.get("result_type",result_type)].replace("K",f'{strat2_args.get("bank_multiplier",bank_multiplier_global)}')
    ax.plot_wireframe(X, Y, Z, color=kwargs.get("color","red"),label=local_title,alpha=kwargs.get("alpha", 0.5))

def plot_num_turns(betting_strategy:Callable, **kwargs):
    Z= np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            Z[i][j] = bet_sim(betting_strategy,  win_prob=x[j], num_turns=y[i], c_kelly=kwargs.get("c_kelly",0.5), **kwargs)
    local_title = betting_strategy.__name__ +" "+result_types[kwargs.get("result_type",0)].replace("K",f'{kwargs.get("bank_multiplier",bank_multiplier_global)}')
    local_title+="\n"
    ax.plot_wireframe(X, Y, Z, color=kwargs.get("color","blue"),label=local_title,alpha=kwargs.get("alpha", 0.5))

# plot_func(kelly_bet, color="green", odds=1.01,uncertainty=uncertainty,result_type=4)
# plot_func(kelly_bet, color="blue", odds=1.01,uncertainty=uncertainty,result_type=8, bank_multiplier=1.15,alpha=0.2)
# plot_diff(kelly_bet, kelly_bet, color="blue", strat2_args={"odds":1.01, "uncertainty":uncertainty, "result_type":8,"bank_multiplier":1.15}, strat1_args={"result_type":4})

# result_type = 7
# plot_func(kelly_bet, color="green", odds=2.0,alpha=1.0)
# plot_func(proportion_bet, color="blue",alpha=0.5)
# plot_diff(proportion_bet,kelly_bet, color="purple", strat2_args={"odds":2.0,"result_type":result_type}, strat1_args={"result_type":result_type})
# plot_func(fixed_bet, color="red",result_type=result_type,alpha=0.5)
# plot_func(kelly_bet, color="red", odds=1.01, result_type=7)
# plot_func(kelly_bet, color="green", odds=1.01, result_type=6)


# plot vs num_turns
ax.set_ylabel("Number of Turns")
plot_num_turns(kelly_bet, color="purple", odds=1.01,uncertainty=uncertainty,result_type=6,alpha=1.0)
cur_title = "Variable Turns in Decreasing Samples\n"

ax.set_zlabel("P(event)")

plt.legend()
# plt.title(cur_title + f"{num_turns} turns with {N} samples")
plt.title(cur_title)
plt.show()

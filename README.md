# Kelly Criterion

This is a project to explore the kelly criterion for trading purposes.

After a few dozen interesting simulations, I can say that the stability
of the kelly criterion is suprsing and effective. Let me list a few results.

- Probability of being profitable increases with the number of trades, under all conditions
- You can tune the variance using a fraction of the kelly value
- high win probabilities > 0.7, and a kelly fraction < 0.3, and a high number of iterations make the most reliable, profitable strategies.
- Without even optimizing for miscalculations of the true win percentage, the kelly fraction provides great results
    - Over a large number of trades, even making a terrible guess can be profitable

These conclusions create an incredibly complelling reason to use the kelly criterion in all types of liquid markets.

Just using this formula instead of making a fixed proportional wager is incredibly effective. It even approximates the best case where your win percentage is 100%. Maybe some variation of this algorithm could be used, where if we want to take a trade, but we scale the actual kelly value by its distance to 100%. This would naturally be a more agressive strategy, but would be a middle ground between the high returns of fixed proportional bets and the more conservative kelly value betting.



## Simulation Notes

The simulations are highly customizable with kargs. We can even use plot_diff to get the probability of making between Cx and Kx. Although its pretty extendable and can generate interesting graphs, the kargs approach is quite confusing.

Rather than improve this simulation directly, I would use this to inform how you should make a propper simulation in C++ utilizing multithreading and the inherent speedup.



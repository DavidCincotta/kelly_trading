import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the range of probabilities and odds
p = np.linspace(0.01, 1.0, 100)  # Probability of winning
b = np.linspace(1.01, 10.0, 100)  # Odds

# Create a meshgrid for probabilities and odds
P, B = np.meshgrid(p, b)

# Calculate q, the probability of losing
Q = 1 - P

# Calculate the fraction of the bankroll to bet according to the Kelly Criterion
F = (B * P - Q) / B
# take the max of 0 and F
F = np.maximum(0, F)

# Plotting
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, B, F, cmap='viridis', edgecolor='none') # type: ignore
ax.set_xlabel('Probability of Winning')
ax.set_ylabel('Odds')
ax.set_zlabel('Fraction to Bet') # type: ignore
ax.set_title('Optimal Fraction to Bet According to the Kelly Criterion')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


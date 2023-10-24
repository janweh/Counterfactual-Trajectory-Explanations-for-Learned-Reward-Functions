from skopt import Optimizer
import numpy as np
import time
t = time.time()
np.random.seed(int(t))
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
import random



# res = gp_minimize(f,                  # the function to minimize
#                   [(-2.0, 2.0)],      # the bounds on each dimension of x
#                   acq_func="EI",      # the acquisition function
#                   n_calls=15,         # the number of evaluations of f
#                   n_random_starts=5,  # the number of random initialization points
#                   noise=0.1**2,       # the noise level (optional)
#                   random_state=1234)   # the random seed

# "x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
# print(res)

# from skopt.plots import plot_convergence
# plot_convergence(res)

opt = Optimizer(
    dimensions=[(0, 2.0), (0, 2.0), (0, 2.0), (0, 2.0), (0, 2.0), (0, 2.0)],
    base_estimator="GP",
    acq_func="EI",

)

def f(x):
    return -(2*x[0] + x[1] + 3*x[2] + x[3] + 2*x[4] + x[5] - 1.5*sum(x))

for i in range(10):
    suggested_x = opt.ask()                      # Get suggested x
    # print(f"Suggested x: {suggested_x}")

    # Here, put your manual procedure to get y=f(x)
    y = f(suggested_x)
    # print(f"Resulting y: {y}")
    
    opt.tell(suggested_x, y)                     # Inform the optimizer of the result

# find the maximum of the function
best_x = opt.Xi[np.argmin(opt.yi)]
print("Best x:", best_x)
best_y = opt.yi[np.argmin(opt.yi)]
print("Best y:", best_y)
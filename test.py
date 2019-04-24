import numpy as np

import matplotlib.pyplot as plt

in_min = 0
in_max = 255
delay_max = 0.1

depth = 10
gamma = 1.5

t = np.arange(0, 255, 1)
f2 = 0

for i in range(depth):
    sigma = (in_max - in_min) / (depth - 2.0) / gamma
    mu = in_min + (i + 1 - 1.5) * ((in_max - in_min) / (depth - 2.0))
    # f = (np.exp(-0.5*((t-mu)/sigma)**2))
    f = (1-np.exp(-0.5*((t-mu)/sigma)**2))*delay_max
    f2 += f * (i+1)
    plt.plot(t, f)

# print(min(f2), max(f2))
plt.figure()
plt.plot(t, f2)
plt.show()

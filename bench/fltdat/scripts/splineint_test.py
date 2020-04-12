import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Create a constant function
no = 128; oo = 0.0; do = 10.0
xo = np.linspace(oo,oo+(no-1)*do,no)

ni = 10; oi = 0.0; di = 10.0
xi = np.zeros(ni)
yi = np.zeros(ni) + 1

# Set endpoints
xi[0] = xo[0]; xi[-1] = xo[-1]

for i in range(1,ni-1):
  xi[i] = xo[np.random.randint(no)]

yi[5] += 0.5
yi[6] += 0.5

xis = sorted(xi)

print(xis)

f = interpolate.interp1d(xis,yi,kind='cubic')

yo = f(xo)

plt.figure(2)
plt.scatter(xis,yi,color='tab:orange')
plt.plot(xo,yo)
plt.ylim([-4,4])

plt.show()

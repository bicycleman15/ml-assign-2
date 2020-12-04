import numpy as np
import matplotlib.pyplot as plt

C = np.array([-5.0, -3.0, 0.0, 0.6989, 1.0])
vals = np.array([0.5736, 0.5736, 0.8808, 0.8828, 0.8824])

plt.plot(C, vals * 100)
plt.xlabel("log values of C")
plt.ylabel("Accuracy on test set")

plt.savefig("plot-c-vs-test.jpg")
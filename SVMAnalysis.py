import numpy as np
import pandas as pd
from sklearn import svm, metrics
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt


plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)

df = pd.read_csv("simdata.csv")
x1 = np.array(df["rf"])
x2 = np.array(df["tp"])
y = np.array(df["Group"])
x = np.vstack((x1, x2)).T

res = .00254

clf = svm.NuSVC(nu=.19, kernel='poly', degree=3, decision_function_shape="ovo").fit(x, y)

x_min, x_max = x[:, 0].min() - .01, x[:, 0].max() + .01
y_min, y_max = x[:, 1].min() - .01, x[:, 1].max() + .01
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))

fig, ax = plt.subplots(figsize=(12, 6))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="jet", alpha=0.3)

labels = [False] * 4
for i in range(x.shape[0]):
    if y[i] == 0:
        plt.scatter(x[i, 0], x[i, 1], c="purple", label="Class A" if not labels[0] else None)
        labels[0] = True
    elif y[i] == 1:
        plt.scatter(x[i, 0], x[i, 1], c="blue", label="Class B" if not labels[1] else None)
        labels[1] = True
    elif y[i] == 2:
        plt.scatter(x[i, 0], x[i, 1], c="goldenrod", label="Class C" if not labels[2] else None)
        labels[2] = True
    elif y[i] == 3:
        plt.scatter(x[i, 0], x[i, 1], c="darkred", label="Class D" if not labels[3] else None)
        labels[3] = True

plt.xlabel("Tau Reduction Factor")
plt.ylabel("Affected Tau Population")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Activity Class", fancybox=True)
plt.title("Therapeutic Activity vs Neural Survival Rate")

accuracy = metrics.accuracy_score(y_true=y, y_pred=clf.predict(x))
print(f"Accuracy: {accuracy}")

plt.show()

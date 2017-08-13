import sys
import numpy as np
from sklearn import svm

def angle(x, y):
    if x < 0:
        return 180 - np.arcsin(y) * 360 / (2 * np.pi)
    else:
        return (0 if y >= 0 else 360) + np.arcsin(y) * 360 / (2 * np.pi)

def inline(inputfile, outputfile):
    data = np.loadtxt(inputfile, delimiter=', ')
    if np.ndim(data) == 1 or len(np.unique(data[:, 0])) == 1:
        loss = 0.5
    else:
        train_x = data[:, 1:]
        train_y = data[:, 0]
        clf = svm.SVC(kernel="linear")
        clf.fit(train_x, train_y)
        ww = clf.coef_[0]
        ww = ww / np.sqrt(sum(np.square(ww)))
        an = angle(ww[0], ww[1])
        loss = min(an / 180, 2 - an / 180)
    with open(outputfile, 'w') as f:
        f.write(str(loss))

if __name__ == "__main__":
    inline(sys.argv[1], sys.argv[2])

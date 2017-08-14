import sys
import numpy as np
from sklearn import svm


def angle(x, y):
    """Compute angle in degrees for points x,y on unit circle"""
    if x < 0:
        return 180 - np.arcsin(y) * 360 / (2 * np.pi)
    else:
        return (0 if y >= 0 else 360) + np.arcsin(y) * 360 / (2 * np.pi)

def compute_loss(inputfile, outputfile):
    """
    Train an SVM based on points given in `inputfile`
    Writes a loss based based on true decision boundary of y = 0 to `outputfile`
    """
    # Read the teaching set in comma-delimited format
    data = np.loadtxt(inputfile, delimiter=', ')
    # If teaching set contains only one label, classify all points with that
    # label, so true loss is 0.5.
    if np.ndim(data) == 1 or len(np.unique(data[:, 0])) == 1:
        loss = 0.5
    else:
        # Assumes the label is in the first column
        train_y = data[:, 0]
        # Assume the remaining columns are features
        train_x = data[:, 1:]
        # Train an SVM
        clf = svm.SVC(kernel="linear")
        clf.fit(train_x, train_y)
        # Extract vector normal to decision boundary
        ww = clf.coef_[0]
        # Normalize
        ww = ww / np.sqrt(sum(np.square(ww)))
        # Return a true loss based on angle with [1,0]
        an = angle(ww[0], ww[1])
        loss = min(an / 180, 2 - an / 180)
    with open(outputfile, 'w') as f:
        f.write(str(loss))

if __name__ == "__main__":
    compute_loss(sys.argv[1], sys.argv[2])

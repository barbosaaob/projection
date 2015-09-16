"""
Projection demo.

Orthogonal projection.
"""

from __future__ import print_function
import projection

try:
    import numpy as np
except ImportError as msg:
    error = ", please install the following packages:\n"
    error += "    NumPy      (http://www.numpy.org)\n"
    raise ImportError(str(msg) + error)


class Ortho(projection.Projection):
    """
    Orthodonal projection.
    """
    def __init__(self, data, data_class, plane=[1, 2]):
        """
        Class initialization.
        """
        dim = data.shape[1]
        assert plane[0] != plane[1] and 0 < plane[0] <= dim and \
            0 < plane[1] <= dim, \
            "*** ERROR (Ortho): plane projection is not well defined"

        projection.Projection.__init__(self, data, data_class, 2)
        self.plane = plane

    def project(self):
        """
        Project method.

        Computes the projection itself.
        """
        assert type(self.data) is np.ndarray, \
            "*** ERROR (Ortho): project input must be of numpy.array type."
        ninst = self.data_ninstances

        proj = np.zeros((ninst, 2))
        proj[:, 0] = self.data[:, self.plane[0] - 1]
        proj[:, 1] = self.data[:, self.plane[1] - 1]

        self.projection = proj


def run():
    import time
    import sys
    print("Loading data set... ", end="")
    sys.stdout.flush()
    data = np.loadtxt("sample-data.data", delimiter=",")
    n, dim = data.shape
    data_class = np.ones(n)
    print("Done.")
    start_time = time.time()
    print("Projecting... ", end="")
    sys.stdout.flush()
    ortho = Ortho(data, data_class, [1, 3])
    ortho.project()
    print("Done. (" + str(time.time() - start_time) + "s)")
    ortho.plot()


if __name__ == "__main__":
    run()

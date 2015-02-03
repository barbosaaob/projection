from __future__ import print_function
import numpy as np
import projection


class Ortho(projection.Projection):
    def __init__(self, data, data_class, plane=[1, 2]):
        dim = data.shape[1]
        assert plane[0] != plane[1] and 0 < plane[0] <= dim and \
            0 < plane[1] <= dim, \
            "*** ERROR (Ortho): plane projection is not well defined"

        projection.Projection.__init__(self, data, data_class, 2)
        self.plane = plane

    def project(self):
        assert type(self.data) is np.ndarray, \
            "*** ERROR (Ortho): project input must be of numpy.array type."
        ninst = self.data_ninstances

        proj = np.zeros((ninst, 2))
        proj[:, 0] = self.data[:, self.plane[0] - 1]
        proj[:, 1] = self.data[:, self.plane[1] - 1]

        self.projection = proj


def test():
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
    test()

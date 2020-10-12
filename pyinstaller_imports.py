# This is a junk file which imports a bunch of packages, to make it much more easy to deal with pyinstaller
# to package up all the correct dependencies. There are more proper ways to do this, but this is the easiest

import numpy
import traceback
import numpy.core.multiarray

try:
    import scipy
    import scipy.linalg
    import scipy.linalg.blas
    import scipy.special
    import scipy.optimize
    # import scipy.optimize.line_search
    import sklearn.utils._cython_blas
    import sklearn.neighbors.typedefs
    import sklearn.neighbors.quad_tree
    import sklearn.tree
    import sklearn.tree._utils
except:
    print("==== scipy")
    print(traceback.format_exc())

try:
    import librosa
except:
    print("==== librosa")
    print(traceback.format_exc())

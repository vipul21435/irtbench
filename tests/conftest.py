"""Keep the linear-algebra backend on one thread.

The matrices in these tests are small; on some numpy builds a multi-threaded BLAS spends far
longer handing them to its thread pool than multiplying them, and the suite becomes slow when
the machine is busy. The variables must be set before numpy is first imported.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

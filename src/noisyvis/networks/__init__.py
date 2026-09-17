"""noisyvis.networks — landscape network builders (plan Stage 8).

    lon          BinaryLON: iterated-local-search Local Optima Networks
    colon        BinaryCoLON: constrained LONs with feasibility tracking
    compression  compress_lon_aggregated: merge optima by fitness accuracy

Explicit exports only. `noisyvis.algorithms` must never import this package: from Stage 8 D the network
modules import `noisyvis.algorithms.operators`, and a reverse import would create a cycle.
"""

from .lon import BinaryLON
from .colon import BinaryCoLON
from .compression import compress_lon_aggregated

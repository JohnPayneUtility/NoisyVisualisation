"""The dashboard's callbacks, one module per callback group.

Importing a module registers its callbacks on `noisyvis.dashboard.instance.app`. This package imports
none of them: the entry module imports each group in registration order.
"""

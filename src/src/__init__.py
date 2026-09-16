"""TEMPORARY compatibility package (plan §5.3).

`src` stopped being the package name in Stage 3; the real package is `noisyvis`, and `src/` is now
only the src-layout root on `sys.path`. This nested package exists so the committed Hydra configs,
which still carry `src.*` dotted paths, keep resolving during the migration.

Deliberately empty: it is a parent package, never an API. Populating it would register
`sys.modules['src.problems']` with real content and mask exactly the dynamic-lookup failure the
config-resolution gate exists to catch (plan §5.6).

Removed in Stage 12, once the configs are rewritten to `noisyvis.*`.
"""

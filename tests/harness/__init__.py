"""Stage 1 isolated-run harness.

Submodules:
    run_isolated -- runs a real entry-point script in a throwaway temp root
    fence        -- audit hook that blocks writes into /workspace
    extract      -- pulls the compared scientific fields out of a finished run
    canonical    -- order-stable JSON encoding and first-difference reporting
"""

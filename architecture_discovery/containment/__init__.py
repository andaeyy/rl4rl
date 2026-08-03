"""Fail-closed containment policy for generated architecture candidates.

The package deliberately separates capability *detection* from evidence that a
boundary was actually enforced.  Importing this package does not make the
existing Python candidate worker a sandbox.  Public types live in
``containment.audit`` and ``containment.policy``; keeping this initializer
side-effect free also makes ``python -m containment.audit`` deterministic.
"""

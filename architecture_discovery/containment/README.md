# Candidate containment boundary

This package reports what the current process can detect and what an external
runner has actually proven. It does not turn the existing Python worker into a
sandbox.

Run the local audit with:

```bash
.venv/bin/python -m containment.audit
```

Exit status `2` means strong containment is not proven. This is the expected
result until an OS-level runner has passed every artifact-bound adversarial
test in `REQUIRED_ADVERSARIAL_TESTS`. Discovering `sandbox-exec`, `resource`, a
sanitized environment, or a Python socket guard is recorded as detected or not
proven. None of those observations alone satisfies the scientific gate.

`BoundaryAttestation` is a schema for evidence produced by a trusted runner.
It is not a cryptographic verifier. Study orchestration must authenticate the
report and bind it to the exact candidate artifact before setting
`authenticated_by_trusted_runner=True`.

The policy has two candidate formats:

- `arbitrary_python` requires all strong OS controls to be proven.
- `architecture_ir` requires typed validation and an evaluator-owned trusted
  interpreter. Post-execution qualification also requires runtime validity
  evidence.

The current repository still executes arbitrary generated Python and therefore
must remain blocked from the scientific lane until either the OS boundary is
proven or candidate generation migrates to the declarative IR.

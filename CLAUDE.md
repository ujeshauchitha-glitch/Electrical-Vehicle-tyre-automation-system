# EV Tyre Intelligence

## Goal

Build an engineering system that estimates the state of individual EV tyres using signals and variables that the vehicle already measures or calculates.

The long-term objective is:

EV telemetry
→ physical observables
→ tyre-state estimation
→ multi-variable fusion
→ temporal degradation estimation
→ degradation-rate prediction

The system should ultimately estimate quantities such as:

- tread depth
- tyre pressure
- tyre wear/degradation rate
- abnormal wear
- relevant tyre/alignment states where observability permits

## Current project stage

We are currently developing and validating the core algorithm.

The initial implementation may be simulation-based. Simulation results must never be presented as real-world validation.

## Architecture

Keep the system separated into:

1. Inputs / EV telemetry
2. Physical models
3. Observable extraction
4. Individual estimators
5. Sensor/state fusion
6. Temporal state tracking
7. Degradation-rate estimation
8. Validation
9. Demo / presentation layer

Avoid putting the entire system into one large Python file.

## Scientific requirements

- Do not invent experimental results.
- Do not claim a variable is observable if the available measurements cannot identify it.
- Keep model assumptions explicit.
- Distinguish measured, simulated, fitted and assumed parameters.
- Preserve uncertainty estimates.
- Test estimators against known ground truth.
- Perform ablation tests where useful.
- Do not manipulate the ground truth to improve performance.
- Clearly identify equations that require real-world validation.

## Existing research prototype

An earlier prototype called `ev_tyre_fusion.py` exists outside this repository and may be imported later.

When it is added to `legacy/`, treat it as reference material.

Do not assume its architecture or equations are correct.
Do not modify it unless explicitly instructed.

## Engineering requirements

- Python for the core research implementation.
- Prefer NumPy and standard scientific Python tools unless another dependency is justified.
- Keep modules small and testable.
- Prefer clear variable and function names.
- Avoid unnecessary abstractions.
- Avoid unnecessary comments.
- Do not create comments that merely restate code.
- Use docstrings only where they provide useful interface or scientific information.
- Write tests for important mathematical behaviour.
- Keep experiments reproducible with explicit random seeds.

## Git requirements

Do not make destructive changes to existing work.

Use descriptive commits.

Do not commit generated datasets, credentials, API keys or large build artifacts unless explicitly required.

## Development workflow

Before implementing a major feature:

1. Understand the existing architecture.
2. Identify assumptions and dependencies.
3. Define the expected inputs and outputs.
4. Implement the smallest useful version.
5. Test it.
6. Validate against known behaviour.
7. Compare against previous implementations where applicable.

The human developer makes final architectural decisions.

AI-generated code must be reviewed rather than blindly accepted.
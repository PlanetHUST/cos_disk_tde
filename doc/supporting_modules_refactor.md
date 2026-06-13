# Supporting Module Refactor

## Scope

This pass consolidates the duplicate simulation drivers and makes physical
units explicit in the supporting modules.

## Authoritative Driver

`test.py` contains the current unified-parallel implementation. `main.py` is
the supported command entry point and re-exports the `test.py` API. This keeps
existing `test.py` users working while preventing the older `main.py`
implementation from drifting independently.

## Unit Convention

The internal convention is cgs:

- mass: grams
- length: centimeters
- time: seconds
- velocity: centimeters per second
- surface density: grams per square centimeter
- volume density: grams per cubic centimeter

Configuration masses are in solar masses and AGN lifetimes are in Myr.
Dimensionless quantities include eccentricity, Toomre Q, disk aspect ratio,
optical depth, and `cos(omega)`.

## Compatibility

Historical public names such as `M_SMBH`, `AGNlifetime`, `initial_mass`,
`t_migration`, `TII`, and `TIII` remain available. Their units are now stated
in docstrings and type contracts.

## Numerical Consequence

`disk_updated.py` previously defined approximate local values for `G`, `c`,
solar mass, parsec, and Myr. It now uses `constants.py`, removing that
cross-module inconsistency. Because the stable disk edge is selected by a
Toomre-Q threshold, small constant changes can move the selected radial grid
index. For `M_SMBH = 1e7 Msun` and `Mdot = 1.44e25 g/s`, the refactored model
returns an outer stable radius of approximately `2.374e16 cm`.

No disk, damping, migration, relaxation, IMF, or stellar-lifetime formula was
intentionally changed.

## Validation Commands

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
python main.py 1 1 /tmp/tde_dyn_smoke 1
```

The last command is a full stochastic simulation and is substantially more
expensive than the unit checks.

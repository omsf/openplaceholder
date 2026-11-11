# Developer notes

## Dependency support

fluvial follows [SPEC0](https://scientific-python.org/specs/spec-0000/).

## Environment management

fluvial uses pixi for management of the development environment.
Users of fluvial should not be concerned with this, as they should install from source, pypi, and conda-forge (as they become available).

## Running tests

Tests are implemented with the `pytest` package.

```
pixi run -e dev pytest src/
```

## Code quality checks

Formatting is enforced through isort and black, which can be applied with their respective commands.

```
pixi run -e dev black src/
pixi run -e dev isort src/
```

Type checking is performed with mypy.

```
pixi run -e dev mypy src/
```

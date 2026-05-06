# Contributing to Numba-CUDA-MLIR

If you are interested in contributing to Numba-CUDA-MLIR, your contributions
will fall into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/numba-cuda-mlir/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The Numba-CUDA-MLIR team will evaluate the issues and triage them. If you
      believe the issue needs priority attention comment on the issue to notify
      the team.
2. You want to propose a new feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.


## Code contributions

### Your first issue

1. Read the project's [INSTALL.md](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/INSTALL.md)
    to learn how to setup the development environment
2. Find an issue to work on.
3. Comment on the issue saying you are going to work on it.
4. Code! Make sure to update unit tests!
   - See the [debugging guidance](#debugging) to aid you during development.
5. Commit your changes.
   - Ensure that your commits are appropriately [signed-off](#signing-your-work).
   - Ensure that commits will pass the [pre-commit checks](#pre-commit-hooks).
6. When done, [create your pull request](https://github.com/NVIDIA/numba-cuda-mlir/compare)
7. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed.
8. Wait for other developers to review your code and update code as needed.
9. Once reviewed and approved, a Numba-CUDA-MLIR developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!


## Pre-commit hooks

We use [pre-commit hooks](https://pre-commit.com/) for formatting and basic linting that should
be applied to every commit.
They can be installed with:

```
pip install -e '.[dev]'
pre-commit install
```

Then, every commit will be formatted and linted automatically.


## Debugging

To dump Numba IR and MLIR to stderr before the MLIR-to-NVVM pipeline, enable `dump` in the `@cuda.jit()` decorator options, e.g. `@cuda.jit(dump=True)`.
To print the full list of available debug options, enable `help` in the `@cuda.jit()` decorator options, e.g. `@cuda.jit(help=True)`.



## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

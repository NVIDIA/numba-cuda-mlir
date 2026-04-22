# __NVIDIA_OSS__ Standard Repo Template

This README file is from the NVIDIA_OSS standard repo template of [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file). It provides a list of files in the PLC-OSS-Template and guidelines on how to use (clone and customize) them.

**Upon completing the customization for the project repo, the repo admin should replace this README template with the project specific README file.**

- Files (org-wide templates in the NVIDIA .github org repo; per-repo overrides allowed) in [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file)

   - Root 
     - README.md skeleton (CTA + Quickstart + Support/Security/Governance links) 
     - LICENSE (Apache 2.0 by default)
        - For other licenses, see the [Confluence page](https://confluence.nvidia.com/pages/viewpage.action?pageId=788418816) for other licenses
        - CLA.md file (delete if not using MIT or BSD licenses)
     - CODE_OF_CONDUCT.md 
     - SECURITY.md (vuln reporting path) 
     - CONTRIBUTING.md (base; repo can add specifics)
     - SUPPORT.md (Support levels/channels)
     - GOVERNANCE.md (baseline; repo may extend)
     - CITATION.md (for projects that need citation)

   - .github/ 
     - ISSUE_TEMPLATE/ (<https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository>)
       - bug.yml, feature.yml, task.yml, config.yml 
     - PULL_REQUEST_TEMPLATE.md (<https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository>)
     - workflows/
     - Note: workflow-templates/ for starter workflows should live in the org-level .github repo, not per-repo

   - Repo-specific (not org-template, maintained by the team)
     - CODEOWNERS (place at .github/CODEOWNERS or repo root)
     - CHANGELOG.md (or RELEASE.md) 
     - ROADMAP.md 
     - MAINTAINERS.md 
     - NOTICE or THIRD_PARTY_NOTICES / THIRD_PARTY_LICENSES (dependency specific)
     - Build/package files (CMake, pyproject, Dockerfile, etc.)

   - Recommended structure and hygiene
     - docs/
     - examples/
     - tests/
     - scripts/
     - Container/dev env: Dockerfile, docker/, .devcontainer/ (optional)
     - Build/package (language-specific):
       - Python: pyproject.toml, setup.cfg/setup.py, requirements.txt, environment.yml
       - C++: CMakeLists.txt, cmake/, vcpkg.json
     - Repo hygiene: .gitignore, .gitattributes, .editorconfig, .pre-commit-config.yaml, .clang-format


## Usage of [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file) for NEW NVIDIA OSS repos

1. Clone the [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template?tab=readme-ov-file)
2. Find/replace all in the clone of `___PROJECT___` and `__PROJECT_NAME__` with the name of the specific project.
3. Inspect all files to make sure all replacements work and update text as needed


**What you can reuse immediately**
- CODE_OF_CONDUCT.md
- SECURITY.md
- CONTRIBUTING.md (base)
- .github/ISSUE_TEMPLATE/.yml (bug/feature/task + config.yml)
- .github/PULL_REQUEST_TEMPLATE.md
- Reusable workflows 

**What you must customize per repo**
- README.md: copy the skeleton and fill in product-specific details (Quickstart, Requirements, Usage, Support level, links)
- LICENSE: check file is correct, update year, consult Confluence for alternatives https://confluence.nvidia.com/pages/viewpage.action?pageId=788418816, add CLA.md only if your license/process requires it
- CODEOWNERS: replace <TEAM> with your GitHub team handle(s). Place at .github/CODEOWNERS (or repo root)
- MAINTAINERS.md: list maintainers names/roles, escalation path
- CHANGELOG.md (or RELEASE.md): track releases/changes
- SUPPORT.md: Update for your project
- ROADMAP.md (optional): upcoming milestones
- NOTICE / THIRD_PARTY_NOTICES (if you ship third‑party content)
- Build/package files (CMake/pyproject/Dockerfile/etc.), tests/, docs/, examples/, scripts/ as appropriate
- Workflows: Edit if you need custom behavior 


4. Change git origin to point to new repo and push
5. Remove the line break below and everything above it

## Usage for existing NVIDIA OSS repos

1. Follow the steps above, but add the files to your existing repo and merge

<!-- REMOVE THE LINE BELOW AND EVERYTHING ABOVE -->
-----------------------------------------
# [Project Title]
One-sentence value proposition for users. Who is it for, and why it matters. 

# Overview
What the project does? Why the project is useful?
Provide a brief overview, highlighting key features or problem-solving capabilities.

# Getting Started
Guide users on how they can get started with the project. This should include basic installation step, quick-start examples 
```bash
# Option A: Package manager (pip/conda/npm/etc.)
<copy-paste install>

# Option B: Container
docker run <image> <args>

# Verify (hello world)
<one-liner or ~10-line example>
```
# Requirements
Include a list of pre-requisites. 
- OS/Arch: <summary or link to full matrix>
- Runtime/Compiler: <versions>
- GPU/Drivers (if applicable): CUDA <ver>, driver <ver>, etc.

# Usage
```bash
# Minimal runnable snippet (≤20 lines)
<code>
```
- More examples/tutorials: <link>
- API reference: <link>

# Performance (Optional)
Summary of benchmarks; link to detailed results and hardware used.

## Releases & Roadmap 
- Releases/Changelog: <link>
- (Optional) Next milestones or link to `ROADMAP.md`.
  
# Contribution Guidelines
- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Development quickstart (build/test):
```bash
<clone> && <deps> && <build/test>
```
## Governance & Maintainers
- Governance: `GOVERNANCE.md`
- Maintainers: <team/handles>
- Labeling/triage policy: <link>

## Security
- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.

## Support
- Level: <Experimental | Maintained | Stable>
- How to get help: Issues/Discussions/<channel link>
- Response expectations (if any).

# Community
Provide the channel for community communications.

# References
Provide a list of related references

# License
This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
- License: <link>

-----------------------------------------
The internal README is below, it will be consolidated into an amenable format to the above soon.

# cuSIMT - CUDA-like Programming Model for Python

🚧 This project is still in the prototype phase. 🚧

cuSIMT provides a low-level programming model similar to CUDA C++ in Python.
The main goals of the project are:

1. Do not inhibit experts
1. Interoperate well with existing programming models

cuSIMT is built on MLIR. We don't use any downstream dialects.
This is not a wrapper around the MLIR Python bindings, however;
user programs look like regular Python, and users do not need
to be compiler experts to use it.

## Prerequisites

- Python >= 3.12
- NVIDIA GPU with a compatible driver (CUDA 12.2+ or 13.x)
    - CUDA Toolkit is **not** required at build time (cuSIMT uses a driver API
      shim header).
    - CUDA Toolkit components (ex: nvJitLink, libNVVM) can be installed via pip (Linux/Windows), conda (Linux/Windows), or any system package manager (Linux).
    - At runtime, CUDA driver and Toolkit components are dynamically loaded.
    - Set `CUDA_HOME` if you need libdevice linking for older paths.

The pinned LLVM commit is in [`ci/llvm-version.env`](ci/llvm-version.env).

## Installation

### Option 1: Pre-built wheel (fastest)

CI publishes wheels to the internal Artifactory PyPI repo on every MR pipeline.
No LLVM, cmake, or CUDA Toolkit needed:

```shell
pip install cusimt[cu13] \
  --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-cuda-python-pypi-local/simple/
```

Replace `cu13` with `cu12` for CUDA 12.x environments.

### Option 2: Editable install with cached LLVM (recommended for development)

CI caches pre-built LLVM tarballs on Artifactory. You can download them
instead of building LLVM from scratch (~1 hour depending on the machine):

1. Compute the cache keys and download from Artifactory (requires
   corpnet/VPN and NVIDIA credentials):
```shell
# Run in a subshell to avoid ci/llvm-cache.sh's "set -e" affecting
# your interactive session:
(
  source ci/llvm-version.env
  source ci/llvm-cache.sh

  MODERN_KEY=$(cache_key "llvm-modern" "$LLVM_MODERN_COMMIT" ci/build-llvm-modern.sh)
  LLVM7_KEY=$(cache_key "llvm7" "$LLVM7_TAG" ci/build-llvm7.sh)

  ARTIF="https://artifactory.nvidia.com/artifactory/sw-cuda-python-generic-local/cusimt/llvm-cache"

  # Use your NVIDIA password when prompted
  wget --user "$USER" --ask-password -O- "$ARTIF/$MODERN_KEY.tar.gz" | tar xzf -
  wget --user "$USER" --ask-password -O- "$ARTIF/$LLVM7_KEY.tar.gz" | tar xzf -
)
```
This produces `llvm-modern-install/` and `llvm7-install/` directories.

2. Create a venv and install cuSIMT in editable mode
```shell
# Using a conda env is also fine
python3 -m venv cusimt-env && source cusimt-env/bin/activate

MLIR_DIR=$PWD/llvm-modern-install/lib/cmake/mlir \
LIBLLVM7=$PWD/llvm7-install/lib/libLLVM-7.so \
  pip install -e '.[cu13,dev]'
```

`setup.py` detects `MLIR_DIR` and `LIBLLVM7` and automatically bundles the
MLIR Python bindings, `libMLIRPythonCAPI.so`, `libLLVM-7.so`, and
`libMLIRToNVVM70.so` into the package.

### Option 3: Build LLVM from source

If you need to modify LLVM/MLIR or the cache doesn't have your platform:

```shell
# Requires sccache for compiler caching (build-time deps are listed
# in pyproject.toml and handled automatically by pip)
pip install sccache   # or: conda install -c conda-forge sccache

# Build modern LLVM + MLIR (uses ci/llvm-version.env for the commit)
ci/build-llvm-modern.sh    # produces llvm-modern-install/

# Build LLVM 7 shared library
ci/build-llvm7.sh          # produces llvm7-install/

# Then install cuSIMT as in Option 2:
MLIR_DIR=$PWD/llvm-modern-install/lib/cmake/mlir \
LIBLLVM7=$PWD/llvm7-install/lib/libLLVM-7.so \
  pip install -e '.[cu13,dev]'
```

See [docs/install-llvm.md](docs/install-llvm.md) for more details.

### NVVM70 path (pre-Blackwell GPUs)

For pre-Blackwell GPUs (< sm_100), cuSIMT uses the mlir-nvvm70 translator
which rebuilds LLVM 7 IR (based on LLVM 7.0.1) from MLIR for the LLVM 7
dialect of NVVM IR. When installed via `MLIR_DIR` + `LIBLLVM7` (Options 2/3),
this is set up automatically.

The NVVM70 path is auto-selected when the target GPU is below sm_100.
See `cext/mlir-nvvm70/README.md` for details on the translator.

## Testing

Our tests are placed in the `tests` directory.
Using `pytest` from the project's root directory after installing cuSIMT will
run our tests. `pytest-xdist` is installed with our testing dependencies, so
they may be run in parallel with:

```
pytest -n 4
```

Note that tests can fail when many threads are used due to the GPU running out of
memory; we re-run tests that fail due to GPU out-of-memory errors by default.
All other errors are reported as test failures.

When testing a fix for a test failure, it is often useful to rerun only the tests
that failed in the previous run; this can be done with `pytest --last-failed` (or `pytest --lf`
for short).

## Run benchmarks and compare performance against numba-cuda (requires NCU)
```
CUSIMT_SKIP_REDIRECTOR=1 pytest tests/benchmarks/ --benchmark -s
```

## Pre-commit hooks

We use [pre-commit hooks](https://pre-commit.com/) for formatting and basic linting that should
be applied to every commit.
They can be installed with:

```
pip install -e '.[test]'
pre-commit install
```

Then, every commit will be formatted and linted automatically.

## Debugging

To dump Numba IR and MLIR to stderr before MLIR-to-NVVM pipeline, enable `dump` in `@cusimt.jit()` decorator options. i.e., use `@cusimt.jit(dump=True)`.
To print full list of available debug options, enable `help` in `@cusimt.jit()` decorator options. i.e., use `@cusimt.jit(help=True)`.

## Benchmark

See [our benchmarking documentation](docs/benchmarks.md) for more details.

## Licensing

cuSIMT is distributed under the [Apache License 2.0](LICENSE).

It incorporates the following third-party projects, each retained under its
original license:

1. [numba-cuda](https://github.com/NVIDIA/numba-cuda) — [BSD 2-Clause License](THIRD-PARTY-LICENSES)
2. [cloudpickle](https://github.com/cloudpipe/cloudpickle) — [BSD 3-Clause License](THIRD-PARTY-LICENSES)
3. [appdirs](https://github.com/ActiveState/appdirs) — [MIT License](THIRD-PARTY-LICENSES)
4. [LLVM Project / EUDSL](https://github.com/llvm/llvm-project) — [Apache License 2.0 WITH LLVM-exception](THIRD-PARTY-LICENSES)
5. [DLPack](https://github.com/dmlc/dlpack) — [Apache License 2.0](THIRD-PARTY-LICENSES)

See [`NOTICE`](NOTICE) for the full attribution map and per-component locations
in this repository, and [`THIRD-PARTY-LICENSES`](THIRD-PARTY-LICENSES) for the
verbatim upstream license texts.

Contributions are accepted under the terms described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
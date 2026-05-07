# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Numba CUDA MLIR"
copyright = "2012-2024 Anaconda Inc. 2024-2026, NVIDIA Corporation."
author = "NVIDIA Corporation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["numpydoc", "sphinx.ext.intersphinx", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "llvmlite": ("https://llvmlite.readthedocs.io/en/latest/", None),
    "numba": ("https://numba.readthedocs.io/en/latest/", None),
}

# To prevent autosummary warnings
numpydoc_show_class_members = False

autodoc_typehints = "none"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/numba-green-icon-rgb.svg"
html_show_sphinx = False


release = os.environ.get("SPHINX_NUMBA_CUDA_MLIR_VER", "0.0.0")

# Wires up the version switcher from versions.json
html_theme_options = {
    "switcher": {
        "json_url": "https://nvidia.github.io/numba-cuda-mlir/versions.json",
        "version_match": release,
    },
    "navbar_center": [
        "version-switcher",
        "navbar-nav",
    ],
}

# Add warning to docs built from main
if int(os.environ.get("BUILD_LATEST", 0)):
    html_theme_options["announcement"] = (
        "Warning: This documentation is built from the development branch\!"
    )

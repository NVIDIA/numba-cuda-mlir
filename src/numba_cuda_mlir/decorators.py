# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable
from collections.abc import Callable as CallableABC
from numba_cuda_mlir import typing
from io import StringIO
from tabulate import tabulate
from numba_cuda_mlir.numba_cuda.core import sigutils
from numba_cuda_mlir.numba_cuda import types as numba_types
from numba_cuda_mlir.numba_cuda.core.typeinfer import register_dispatcher
import inspect
import sys
from textwrap import dedent
from dataclasses import dataclass
from numba_cuda_mlir.tools import get_gpu_compute_capability, format_arch


@dataclass
class MLIRJITOption:
    name: str
    types: type | tuple[type, ...]
    default_value: Any
    help: str
    hidden: bool = False
    extra_verification: Callable[[Any, dict[str, Any]], str | None] | None = None

    def verify_value(self, value: Any, targetoptions: dict[str, Any]) -> str | None:
        if not isinstance(value, self.types):
            return f"Expected {self.name} to be of type {self.types}, got {value}"
        if self.extra_verification is not None:
            return self.extra_verification(value, targetoptions)
        return None


def _verify_opt_level(value: Any, targetoptions: dict[str, Any]) -> str | None:
    if value < 0 or value > 3:
        return f"Expected opt_level to be between 0 and 3, got {value}"
    return None


def _verify_chip(value: Any, targetoptions: dict[str, Any]) -> str | None:
    if not value.startswith("sm_"):
        return f"Expected chip to start with 'sm_', got {value}"
    return None


def _verify_inline(value: Any, targetoptions: dict[str, Any]) -> str | None:
    if isinstance(value, bool) or callable(value):
        return None
    options = ["always", "never", "auto"]
    if value not in options:
        return f"Expected inline to be one of {options}, True, False, or a callable, got {value}"
    return None


def _verify_abi(value: Any, targetoptions: dict[str, Any]) -> str | None:
    options = ["c", "numba"]
    if value not in options:
        raise NotImplementedError(f"Unsupported ABI: {value}")
    return None


def _verify_shared_memory_carveout(value: Any, targetoptions: dict[str, Any]) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        valid_strings = ["default", "maxl1", "maxshared"]
        if value.lower() not in valid_strings:
            raise ValueError(
                f"Invalid carveout value: {value}. Must be -1 to 100 or one of {valid_strings}"
            )
        return None
    if isinstance(value, int):
        if not (-1 <= value <= 100):
            raise ValueError("Carveout must be between -1 and 100")
        return None
    raise TypeError(f"shared_memory_carveout must be str or int, got {type(value).__name__}")


def _verify_launch_bounds(value: Any, targetoptions: dict[str, Any]) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            return f"launch_bounds must be positive, got {value}"
        return None
    if isinstance(value, tuple):
        if len(value) > 3:
            return f"Got {len(value)} launch bounds: expected at most 3 (max_threads_per_block, min_blocks_per_sm, max_cluster_rank)"
        if len(value) == 0:
            return "launch_bounds tuple cannot be empty"
        for i, v in enumerate(value):
            if not isinstance(v, int) or v <= 0:
                return f"launch_bounds[{i}] must be a positive integer, got {v}"
        return None
    return f"launch_bounds must be int, tuple, or None, got {type(value).__name__}"


def _get_schema() -> tuple[MLIRJITOption, ...]:
    schema = (
        MLIRJITOption(
            name="opt_level",
            types=int,
            default_value=3,
            help="Optimization level",
            extra_verification=_verify_opt_level,
        ),
        MLIRJITOption(
            name="capi",
            types=(bool, str),
            default_value=False,
            help="Use the function's name as the symbol name in the generated cubin. False means use the mangled name, true means use the function's name, and a string means use the string as the symbol name.",
        ),
        MLIRJITOption(
            name="inline",
            types=(str, bool, CallableABC),
            default_value="always",
            help="Inline strategy",
            extra_verification=_verify_inline,
        ),
        MLIRJITOption(
            name="print_before_all",
            types=bool,
            default_value=False,
            help="Print MLIR to stderr before every optimization pass",
        ),
        MLIRJITOption(
            name="print_after_all",
            types=bool,
            default_value=False,
            help="Print MLIR to stderr after every optimization pass",
        ),
        MLIRJITOption(
            name="chip",
            types=str,
            default_value=get_gpu_compute_capability(),
            help="GPU compute capability used for compilation",
            extra_verification=_verify_chip,
        ),
        MLIRJITOption(
            name="features",
            types=str,
            default_value=None,
            help="Additional PTX features to enable (for example, '+ptx80')",
        ),
        MLIRJITOption(
            name="ptxas_options",
            types=str,
            default_value=None,
            help="Additional options to pass to ptxas",
        ),
        MLIRJITOption(
            name="dump",
            types=bool,
            default_value=False,
            help="Dump Numba IR and MLIR to stderr before compilation",
        ),
        MLIRJITOption(
            name="dump_numba_ir",
            types=bool,
            default_value=False,
            help="Dump Numba IR to stderr before compilation",
        ),
        MLIRJITOption(
            name="dump_mlir",
            types=bool,
            default_value=False,
            help="Dump MLIR to stderr before compilation",
        ),
        MLIRJITOption(
            name="dump_cubin",
            types=bool,
            default_value=False,
            help="Dump cubin to stderr after compilation",
        ),
        MLIRJITOption(
            name="dump_llvmir",
            types=bool,
            default_value=False,
            help="Dump LLVM IR to stderr before libnvvm compilation",
        ),
        MLIRJITOption(
            name="dump_ptx",
            types=bool,
            default_value=False,
            help="Dump PTX to stderr before compilation",
        ),
        MLIRJITOption(
            name="dump_ast",
            types=bool,
            default_value=False,
            help="Dump Python AST before/after transformations",
        ),
        MLIRJITOption(
            name="dump_ast_after_all",
            types=bool,
            default_value=False,
            help="Dump Python AST after each transformation pass",
        ),
        MLIRJITOption(
            name="experimental_ast_transforms",
            types=bool,
            default_value=False,
            help="Enable experimental AST transforms (consteval, loop unrolling)",
        ),
        MLIRJITOption(
            name="debug",
            types=bool,
            default_value=False,
            help="Emit debug information",
        ),
        MLIRJITOption(
            name="fastmath",
            types=bool,
            default_value=False,
            help="Use faster approximations for floating-point arithmetic",
        ),
        MLIRJITOption(
            name="device",
            types=bool,
            default_value=False,
            help="Device-only functions can only be called by a kernel or other device functions",
        ),
        MLIRJITOption(
            name="link",
            types=list,
            default_value=[],
            help="List of libraries to link against",
        ),
        MLIRJITOption(
            name="lto",
            types=bool,
            default_value=False,
            help="Enable link time optimization",
        ),
        MLIRJITOption(
            name="no_cpython_wrapper",
            types=bool,
            default_value=True,
            help="Only used by Numba",
            hidden=True,
        ),
        MLIRJITOption(
            name="nopython",
            types=bool,
            default_value=True,
            help="Only used by Numba",
            hidden=True,
        ),
        MLIRJITOption(
            name="_nrt",
            types=bool,
            default_value=False,
            help="Only used by Numba",
            hidden=True,
        ),
        MLIRJITOption(
            name="signature",
            types=(tuple, list, str),
            default_value=None,
            help="Function signature to compile for. Setting to 'infer' will infer the signature from the type annotations of the arguments",
        ),
        MLIRJITOption(
            name="fast_math",
            types=(bool, type(None)),
            default_value=None,
            help="Alias for fastmath",
            hidden=True,
        ),
        MLIRJITOption(
            name="opt",
            types=(bool, int, type(None)),
            default_value=None,
            help="Enable optimization (numba compatibility, maps to opt_level)",
            hidden=True,
        ),
        MLIRJITOption(
            name="forceinline",
            types=bool,
            default_value=False,
            help="Force inlining of device functions",
            hidden=True,
        ),
        MLIRJITOption(
            name="lineinfo",
            types=bool,
            default_value=False,
            help="Emit line number information for profiling",
        ),
        MLIRJITOption(
            name="max_registers",
            types=int,
            default_value=None,
            help="Maximum number of registers per thread",
        ),
        MLIRJITOption(
            name="launch_bounds",
            types=(tuple, int, type(None)),
            default_value=None,
            help="Launch bounds (max_threads_per_block, min_blocks_per_sm) or just max_threads_per_block as int",
            extra_verification=_verify_launch_bounds,
        ),
        MLIRJITOption(
            name="abi",
            types=str,
            default_value="numba",
            help="ABI for the compiled function",
            extra_verification=_verify_abi,
        ),
        MLIRJITOption(
            name="abi_info",
            types=(dict, type(None)),
            default_value=None,
            help="Additional ABI information",
            hidden=True,
        ),
        MLIRJITOption(
            name="cache",
            types=bool,
            default_value=False,
            help="Enable on-disk caching of compiled kernels",
            hidden=True,
        ),
        MLIRJITOption(
            name="cc",
            types=(tuple, type(None)),
            default_value=None,
            help="Compute capability (numba compatibility, maps to chip)",
            hidden=True,
        ),
        MLIRJITOption(
            name="output",
            types=str,
            default_value="ptx",
            help="Output format for compile functions",
            hidden=True,
        ),
        MLIRJITOption(
            name="_dbg_optnone",
            types=bool,
            default_value=False,
            help="Debug option to disable optimizations (numba compatibility)",
            hidden=True,
        ),
        MLIRJITOption(
            name="annotations_as_signatures",
            types=bool,
            default_value=True,
            help="When True, type annotations define the kernel signature and are strictly enforced at decoration time",
            hidden=True,
        ),
        MLIRJITOption(
            name="extensions",
            types=(list, type(None)),
            default_value=None,
            help="List of argument handler extensions with prepare_args method",
        ),
        MLIRJITOption(
            name="shared_memory_carveout",
            types=(str, int, type(None)),
            default_value=None,
            help="Shared memory carveout percentage (-1 to 100, or 'default'/'maxl1'/'maxshared')",
            extra_verification=_verify_shared_memory_carveout,
        ),
        MLIRJITOption(
            name="profile_jit",
            types=(bool, str),
            default_value=False,
            help="Profile compilation time with cProfile. True prints stats to stderr, a string path also saves the .prof file.",
        ),
    )
    return tuple(sorted(schema, key=lambda x: x.name))


def _format_option(option: MLIRJITOption) -> tuple[str, str, str, str] | None:
    if option.hidden:
        return None
    name, types, default_value, help = (
        option.name,
        option.types,
        option.default_value,
        option.help,
    )

    def _format_type(t: type) -> str:
        return t.__name__

    if isinstance(types, tuple):
        types = "|".join(list(map(_format_type, types)))
    else:
        types = _format_type(types)
    return name, types, str(default_value), help


def _target_options_help(why: str | None = None):
    if why:
        why = f"Printing the options because: {why.strip()}"
    else:
        why = ""
    prelude = dedent(
        f"""
    {why}
    The available keyword options for numba_cuda_mlir.jit are:

    """
    )
    formatted_options = filter(None, map(_format_option, _get_schema()))
    return prelude + tabulate(
        headers=["Name", "Types", "Default Value", "Help"],
        tabular_data=formatted_options,
    )


def _extract_signature_from_annotations(func):
    """Extract Numba signature from function type annotations.

    Returns a Numba signature if all parameters have valid type annotations,
    or None if any parameter lacks an annotation (template/lazy mode).
    """
    from numba_cuda_mlir.lowering_utilities.type_conversions import to_numba_type

    sig = inspect.signature(func)
    argtypes = []
    has_unannotated = False

    # A function with no parameters cannot have parameter annotations, so
    # there is no annotation-derived signature to return.
    if not sig.parameters:
        return None

    for param in sig.parameters.values():
        ann = param.annotation
        if ann == inspect.Parameter.empty:
            has_unannotated = True
            argtypes.append(None)
        elif isinstance(ann, numba_types.Type):
            argtypes.append(ann)
        else:
            try:
                argtypes.append(to_numba_type(ann))
            except (TypeError, NotImplementedError):
                # Unrecognized annotation type - treat as template
                has_unannotated = True
                argtypes.append(None)

    # If any parameter is unannotated/unrecognized, use lazy compilation
    if has_unannotated:
        return None

    # Get return type if annotated
    ret_ann = sig.return_annotation
    if ret_ann != inspect.Signature.empty:
        if isinstance(ret_ann, numba_types.Type):
            return_type = ret_ann
        else:
            try:
                return_type = to_numba_type(ret_ann)
            except (TypeError, NotImplementedError):
                return_type = numba_types.none
    else:
        return_type = numba_types.none

    return typing.signature(return_type, *argtypes)


def _get_signatures(func_or_sig):
    if sigutils.is_signature(func_or_sig):
        return [func_or_sig]
    elif isinstance(func_or_sig, list):
        return func_or_sig
    elif callable(func_or_sig) or func_or_sig is None:
        return None
    else:
        raise ValueError(_target_options_help(f"Invalid function or signature: {func_or_sig}."))


def verify_target_options(kws: dict[str, Any]) -> dict[str, Any]:
    targetoptions = kws.copy()

    if "help" in kws:
        raise SystemExit(_target_options_help("help was specified"))

    if "opt" in kws and "opt_level" in kws:
        raise ValueError("Cannot specify both opt and opt_level at the same time.")

    invalid_options = set(kws) - {option.name for option in _get_schema()}
    if invalid_options:
        raise ValueError(
            _target_options_help(f"Got invalid options: {', '.join(invalid_options)}.")
        )

    targetoptions = kws.copy()
    schema = _get_schema()
    for option in schema:
        if option.name not in kws:
            targetoptions[option.name] = option.default_value
        else:
            if not isinstance(targetoptions[option.name], option.types):
                raise TypeError(
                    _target_options_help(
                        f"Expected {option.name} to be of type {option.types}, got {targetoptions[option.name]}"
                    )
                )
            if option.extra_verification is not None:
                error = option.extra_verification(targetoptions[option.name], targetoptions)
                if error:
                    raise ValueError(_target_options_help(f"{option.name}: {error}"))

    # Normalize inline booleans to strings (numba's InlineOptions only accepts
    # "always", "never", or callables -- not True/False).
    if "inline" in kws and isinstance(kws["inline"], bool):
        targetoptions["inline"] = "always" if kws["inline"] else "never"

    # Handle compatibility mappings
    if targetoptions.get("fast_math") is not None:
        targetoptions["fastmath"] = targetoptions["fast_math"]
    if targetoptions.get("opt") is not None:
        targetoptions["opt_level"] = 3 if targetoptions["opt"] else 0

    opt_level = targetoptions.get("opt_level", 3)
    if targetoptions.get("debug") and opt_level > 0:
        raise ValueError(
            "debug=True requires opt_level=0 (ptxas does not support optimized "
            "debugging). Set opt_level=0 or opt=False."
        )

    if targetoptions.get("cc") is not None:
        cc = targetoptions["cc"]
        if isinstance(cc, tuple):
            targetoptions["chip"] = format_arch(cc)

    # When LTO is enabled, output LTOIR instead of PTX
    if targetoptions.get("lto", False):
        targetoptions["output"] = "ltoir"

    return targetoptions


def mlir_jit(func_or_sig=None, **kws):
    """
    numba_cuda_mlir JIT decorator. Use this function to decorate kernels and device functions.
    """
    import warnings

    from numba_cuda_mlir.descriptor import MLIRDispatcher
    from numba_cuda_mlir.numba_cuda.core import config as cuda_config
    from numba_cuda_mlir.numba_cuda.core.errors import NumbaInvalidConfigWarning

    debug = kws.get("debug")
    if debug is None:
        debug = cuda_config.CUDA_DEBUGINFO_DEFAULT != 0
    opt = kws.get("opt")
    if opt is None:
        opt = cuda_config.OPT != 0
    lineinfo = kws.get("lineinfo", False)
    if debug and lineinfo:
        warnings.warn(
            NumbaInvalidConfigWarning(
                "debug and lineinfo are mutually exclusive. Use debug to get "
                "full debug info (this disables some optimizations), or "
                "lineinfo for line info only with code generation unaffected."
            )
        )
    if debug and opt:
        warnings.warn(
            NumbaInvalidConfigWarning(
                "debug=True with opt=True "
                "is not supported by CUDA. This may result in a crash"
                " - set debug=False or opt=False."
            )
        )

    signatures = _get_signatures(func_or_sig)

    targetoptions = verify_target_options(kws)
    annotations_as_signatures = targetoptions.get("annotations_as_signatures", True)

    def _jit_with_signatures(func):
        """JIT when explicit signatures are provided."""
        nonlocal signatures

        # Check for conflicting signature sources
        if annotations_as_signatures:
            annotation_sig = _extract_signature_from_annotations(func)
            if annotation_sig is not None:
                raise TypeError(
                    f"Conflicting signature sources for '{func.__name__}': "
                    f"both an explicit signature and type annotations are provided. "
                    f"Use one or the other:\n"
                    f"  - Remove the explicit signature and rely on annotations, or\n"
                    f"  - Remove annotations and use the explicit signature, or\n"
                    f"  - Set annotations_as_signatures=False to ignore annotations."
                )

        disp = MLIRDispatcher(func, targetoptions=targetoptions)

        if targetoptions.get("cache", False):
            disp.enable_caching()

        if targetoptions.get("signature") == "infer":
            signatures = _get_signatures(func)

        with register_dispatcher(disp):
            for sig in signatures:
                argtypes, restype = sigutils.normalize_signature(sig)
                disp.compile(argtypes)

        disp.disable_compile()
        disp._specialized = True

        return disp

    def _jit_with_annotations(func):
        """JIT using type annotations as the signature if available."""
        sig = _extract_signature_from_annotations(func)

        disp = MLIRDispatcher(func, targetoptions=targetoptions)

        if targetoptions.get("cache", False):
            disp.enable_caching()

        if sig is not None:
            # Annotations present - pre-compile with annotated signature
            argtypes, restype = sigutils.normalize_signature(sig)
            with register_dispatcher(disp):
                disp.compile(argtypes)
            disp.disable_compile()
            disp._specialized = True

        # If no annotations, fall back to lazy compilation
        return disp

    def _jit_lazy(func):
        """JIT with lazy compilation (non-binding mode)."""
        disp = MLIRDispatcher(func, targetoptions=targetoptions)
        if targetoptions.get("cache", False):
            disp.enable_caching()
        return disp

    # If explicit signatures provided, use them
    if signatures is not None:
        return _jit_with_signatures

    # No explicit signatures - check annotations_as_signatures mode
    if func_or_sig is None:
        # @jit() style - return decorator
        if annotations_as_signatures:
            return _jit_with_annotations
        else:
            return _jit_lazy
    else:
        # @jit style - func_or_sig is the function
        if annotations_as_signatures:
            return _jit_with_annotations(func_or_sig)
        else:
            return _jit_lazy(func_or_sig)


def stubgen(out: StringIO = sys.stdout):
    schema = _get_schema()
    from textwrap import dedent, indent
    import numba_cuda_mlir
    from pathlib import Path

    this_file = Path(__file__).relative_to(Path(numba_cuda_mlir.__path__[0]).parent)

    out.write(
        dedent(
            f"""
    # Autogenerated by {this_file}
    # Please do not edit

    from typing import Callable
    from numba_cuda_mlir.descriptor import MLIRDispatcher

    def jit(
        func_or_sig : Callable | str | None = None,
    """
        )
    )

    def _type_name(t: type) -> str:
        return "None" if t is type(None) else t.__name__

    for option in schema:
        if option.hidden:
            continue
        types = option.types
        if isinstance(types, tuple):
            types = " | ".join(map(_type_name, types))
        else:
            types = _type_name(types)

        default = option.default_value
        if isinstance(default, str):
            default = f"'{default}'"
        elif isinstance(default, bool):
            default = "True" if default else "False"
        elif option.types is bool and isinstance(default, int):
            default = "True" if default else "False"

        out.write(
            indent(
                dedent(
                    f"""\
        {option.name}: {types} = {default}, # {option.help}
        """
                ),
                " " * 4,
            )
        )
    docstring = mlir_jit.__doc__
    out.write(
        dedent(
            f"""\
    ) -> MLIRDispatcher:
        '''{indent(docstring, " " * 4)}
        '''
    """
        )
    )

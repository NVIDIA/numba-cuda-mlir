# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Self
from numba_cuda_mlir.tools import get_gpu_compute_capability, format_arch
from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker


class Linker(_Linker):
    def __init__(
        self,
        cc: tuple[int, int] | None = None,
        additional_flags: list[str] | None = None,
        lto: bool | None = None,
        arch: str | None = None,
        verbose: bool = False,
        ftz: bool | None = None,
        prec_div: bool | None = None,
        prec_sqrt: bool | None = None,
        fma: bool | None = None,
        debug: bool = False,
        lineinfo: bool = False,
        optimize_unused_variables: bool = True,
        optimization_level: int = 3,
        ptxas_options: str | None = None,
        max_registers: int | None = None,
    ):
        if cc is None:
            cc = get_gpu_compute_capability(tuple)
        if arch is None:
            arch = format_arch(cc)
        super().__init__(
            cc=cc,
            additional_flags=additional_flags,
            lineinfo=lineinfo,
            lto=lto,
            max_registers=max_registers,
            arch=arch,
            verbose=verbose,
            debug=debug,
            ftz=ftz,
            prec_div=prec_div,
            prec_sqrt=prec_sqrt,
            fma=fma,
            optimize_unused_variables=optimize_unused_variables,
            optimization_level=optimization_level,
            ptxas_options=ptxas_options,
        )

        self._numba_cuda_mlir_temp_ptx_files: list[str] = []
        self._ltoirs: dict[int, bytes] = {}

    def recreate_with_lto(self, lto: bool = True) -> Self:
        """Recreate the linker, re-adding all object codes from raw bytes."""
        existing = list(getattr(self, "_object_codes", []))
        new_linker = Linker(
            cc=self.cc,
            additional_flags=self.additional_flags,
            lto=lto,
            arch=self.arch,
            verbose=self._verbose,
            ftz=self._ftz,
            prec_div=self._prec_div,
            prec_sqrt=self._prec_sqrt,
            fma=self._fma,
            debug=self._debug,
            lineinfo=self.lineinfo,
            optimize_unused_variables=self._optimize_unused_variables,
            optimization_level=self._optimization_level,
            ptxas_options=self._ptxas_options,
            max_registers=self.max_registers,
        )
        for obj in existing:
            code_type = getattr(obj, "code_type", None)
            if code_type == "ptx":
                new_linker.add_ptx(obj.code)
            elif code_type == "cubin":
                new_linker.add_cubin(obj.code)
            elif code_type == "ltoir":
                new_linker.add_ltoir(obj.code)
            else:
                new_linker._object_codes.append(obj)
        new_linker._ltoirs = dict(self._ltoirs)
        return new_linker

    def add_ltoir(self, ltoir: bytes, name: str = "") -> None:
        h = hash(ltoir)
        if h in self._ltoirs:
            return self
        if not self.lto:
            new_linker = self.recreate_with_lto()
            self.__dict__.update(new_linker.__dict__)
        super().add_ltoir(ltoir, name)
        self._ltoirs[h] = ltoir

    def merge_ltoirs_from(self, other: "Linker") -> None:
        for ltoir in other._ltoirs.values():
            self.add_ltoir(ltoir)

    def _get_linker_options(self, ptx):
        return super()._get_linker_options(ptx)

..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause


Device management
=================

For multi-GPU machines, users may want to select which GPU to use.
By default the CUDA driver selects the fastest GPU as the device 0,
which is the default device used by Numba CUDA MLIR.

The features introduced on this page are generally not of interest
unless working with systems hosting/offering more than one CUDA-capable GPU.

Device Selection
----------------

Select the device on which subsequent CUDA operations should run:

::

    from numba_cuda_mlir import cuda
    cuda.select_device(0)

Users can select another device and reuse the same kernel dispatcher:

::

    import numpy as np

    @cuda.jit
    def increment(out, value):
        out[0] = value + 1

    configured = increment[1, 1]
    for device_id in (0, 1, 0):  # assuming we have 2 GPUs
        with cuda.gpus[device_id]:
            out = cuda.device_array(1, np.int32)
            configured(out, np.int32(40))
            assert out.copy_to_host()[0] == 41

Compilation and launch caches follow the selected device and context lifetime.
Returning to a device reuses its specialization. Configured calls also remain
usable across device selection. Array memory must be accessible from the
selected context, and streams must belong to it. Passing an array does not
change the selected device. Resetting a context invalidates its arrays, streams,
loaded functions, and allocator state.

An explicit ``chip`` option continues to control compilation. An inferred
architecture is resolved for each target without changing the dispatcher's
user-specified options.

Calling ``disable_compile()`` freezes each existing context's compiled
signatures. New contexts and serialized dispatchers inherit the combined
frozen signatures, including literal arguments and launch specializations,
even after the original contexts expire. Compiled modules remain specific
to each context.


.. function:: numba_cuda_mlir.cuda.select_device(device_id)
   :noindex:

   Create a new CUDA context for the selected *device_id*.  *device_id*
   should be the number of the device (starting from 0; the device order
   is determined by the CUDA libraries).  The context is associated with
   the current thread. Only one context per thread is permitted.

   If successful, this function returns a device instance.

   .. XXX document device instances?


.. function:: numba_cuda_mlir.cuda.close
   :noindex:

   Explicitly close all contexts in the current thread.

   .. note::
      Resetting a context discards its cached launch state. With
      ``cuda-core`` 1.1.1, creating a new context afterward can fail with
      ``CUDA_ERROR_CONTEXT_IS_DESTROYED``. Use ``cuda.gpus[device_id]`` to
      switch devices while continuing to use existing dispatchers.

The Device List
===============

The Device List is a list of all the GPUs in the system, and can be indexed to
obtain a context manager that ensures execution on the selected GPU.

.. attribute:: numba_cuda_mlir.cuda.gpus
   :noindex:
.. attribute:: numba_cuda_mlir.numba_cuda.cudadrv.devices.gpus

:py:data:`numba_cuda_mlir.cuda.gpus` is an instance of the ``_DeviceList`` class, from
which the current GPU context can also be retrieved:

.. autoclass:: numba_cuda_mlir.numba_cuda.cudadrv.devices._DeviceList
    :members: current
    :noindex:


Device UUIDs
============

The UUID of a device (equal to that returned by ``nvidia-smi -L``) is available
in the :attr:`uuid <numba_cuda_mlir.cuda.cudadrv.driver.Device.uuid>` attribute
of a CUDA device object.

For example, to obtain the UUID of the current device:

.. code-block:: python

   dev = cuda.current_context().device
   # prints e.g. "GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643"
   print(dev.uuid)

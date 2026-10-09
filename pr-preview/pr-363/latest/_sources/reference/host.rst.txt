..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

CUDA Host API (Deprecated)
==========================

.. warning:: The host API functions are not recommended for use in new code,
   and are provided for backwards compatibility with code written for
   Numba-CUDA. It is recommended that the `cuda.core
   <https://nvidia.github.io/cuda-python/cuda-core/latest/>`_ equivalents are
   used for new code.

Device Management
-----------------

.. note:: See `Devices and execution
   <https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#devices-and-execution>`_
   in the cuda.core documentation for recommended replacement APIs.

Device detection and enquiry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions are available for querying the available hardware:

.. autofunction:: numba.cuda.is_available

.. autofunction:: numba.cuda.detect

Context management
~~~~~~~~~~~~~~~~~~

CUDA Python functions execute within a CUDA context. Each CUDA device in a
system has an associated CUDA context, and Numba presently allows only one context
per thread. For further details on CUDA Contexts, refer to the `CUDA Driver API
Documentation on Context Management
<http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html>`_ and the
`CUDA C Programming Guide Context Documentation
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#context>`_. CUDA Contexts
are instances of the :class:`~numba.cuda.cudadrv.driver.Context` class:

.. autoclass:: numba.cuda.cudadrv.driver.Context
   :members: reset, get_memory_info, push, pop

The following functions can be used to get or select the context:

.. autofunction:: numba.cuda.current_context
.. autofunction:: numba.cuda.require_context

The following functions affect the current context:

.. autofunction:: numba.cuda.synchronize
.. autofunction:: numba.cuda.close

Device management
~~~~~~~~~~~~~~~~~

Numba maintains a list of supported CUDA-capable devices:

.. attribute:: numba.cuda.gpus

   An indexable list of supported CUDA devices. This list is indexed by integer
   device ID.

Alternatively, the current device can be obtained:

.. attribute:: numba.cuda.gpus.current

   The currently-selected device.

Getting a device through :attr:`numba.cuda.gpus` always provides an instance of
:class:`numba.cuda.cudadrv.devices._DeviceContextManager`, which acts as a
context manager for the selected device:

.. autoclass:: numba.cuda.cudadrv.devices._DeviceContextManager

One may also select a context and device or get the current device using the
following three functions:

.. autofunction:: numba.cuda.select_device
.. autofunction:: numba.cuda.get_current_device
.. autofunction:: numba.cuda.list_devices

The :class:`numba.cuda.cudadrv.driver.Device` class can be used to enquire about
the functionality of the selected device:

.. class:: numba.cuda.cudadrv.driver.Device

   The device associated with a particular context.

   .. attribute:: compute_capability

      A tuple, *(major, minor)* indicating the supported compute capability.

   .. attribute:: id

      The integer ID of the device.

   .. attribute:: name

      The name of the device (e.g. "GeForce GTX 970").

   .. attribute:: uuid

      The UUID of the device (e.g. "GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643").

   .. method:: reset

      Delete the context for the device. This will destroy all memory
      allocations, events, and streams created within the context.

   .. attribute:: supports_float16

      Return ``True`` if the device supports float16 operations, ``False``
      otherwise.



.. _events:

Events
~~~~~~

.. note:: See the `Event class
   <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Event.html>`_
   in the cuda.core documentation for recommended replacement APIs.

Events can be used to monitor the progress of execution and to record the
timestamps of specific points being reached. Event creation returns immediately,
and the created event can be queried to determine if it has been reached. For
further information, see the `CUDA C Programming Guide Events section
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#events>`_.

The following functions are used for creating and measuring the time between
events:

.. autofunction:: numba.cuda.event
.. autofunction:: numba.cuda.event_elapsed_time

Events are instances of the :class:`numba.cuda.cudadrv.driver.Event` class:

.. autoclass:: numba.cuda.cudadrv.driver.Event
   :members: query, record, synchronize, wait


.. _streams:

Stream Management
-----------------

.. note:: See the `Stream class
   <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html>`_
   in the cuda.core documentation for recommended replacement APIs.

Streams allow concurrency of execution on a single device within a given
context. Queued work items in the same stream execute sequentially, but work
items in different streams may execute concurrently. Most operations involving a
CUDA device can be performed asynchronously using streams, including data
transfers and kernel execution. For further details on streams, see the `CUDA C
Programming Guide Streams section
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams>`_.

Numba defaults to using the legacy default stream as the default stream. The
per-thread default stream can be made the default stream by setting the
environment variable ``NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM`` to ``1`` (see the
:ref:`CUDA Environment Variables section <numba-envvars-gpu-support>`).
Regardless of this setting, the objects representing the legacy and per-thread
default streams can be constructed using the functions below.

Streams are instances of :class:`numba.cuda.cudadrv.driver.Stream`:

.. autoclass:: numba.cuda.cudadrv.driver.Stream
   :members: synchronize, auto_synchronize, add_callback, async_done

To create a new stream:

.. autofunction:: numba.cuda.stream

To get the default stream:

.. autofunction:: numba.cuda.default_stream

To get the default stream with an explicit choice of whether it is the legacy
or per-thread default stream:

.. autofunction:: numba.cuda.legacy_default_stream

.. autofunction:: numba.cuda.per_thread_default_stream

To construct a Numba ``Stream`` object using a stream allocated elsewhere, the
``external_stream`` function is provided. Note that the lifetime of external
streams must be managed by the user - Numba will not deallocate an external
stream, and the stream must remain valid whilst the Numba ``Stream`` object is
in use.

.. autofunction:: numba.cuda.external_stream

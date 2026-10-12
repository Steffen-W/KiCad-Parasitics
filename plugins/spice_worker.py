"""libngspice runs in a child process, so a hanging or crashing simulation can be killed."""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import sys
import time
from logging.handlers import QueueHandler
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from queue import SimpleQueue
from types import ModuleType
from typing import Callable

SPICE_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 0.1


class SpiceUnavailableError(Exception):
    """Raised when ngspice cannot be started; the message addresses the user."""


class SpiceError(RuntimeError):
    """Raised when an ngspice simulation produces no usable result."""


class SpiceWorker:
    """Runs netlists in a child process that owns libngspice.

    Errors ngspice reports leave the process running. A timeout, an exception
    from `on_wait` or a dead process kills it, and the next call starts a new
    one. Calls must not overlap: the process serves one request at a time.
    """

    def __init__(self, on_wait: Callable[[], None]) -> None:
        """`on_wait` is called repeatedly while waiting for ngspice."""
        self._on_wait = on_wait
        self._child: tuple[BaseProcess, Connection] | None = None

    def measure(self, netlist: list[str], vector: str) -> complex:
        """Run the netlist's analysis and return the single value of `vector`.

        Raises:
            SpiceUnavailableError: the process cannot be started or cannot load
                libngspice within SPICE_TIMEOUT_S
            SpiceError: rejected netlist, no single result value, no answer
                within SPICE_TIMEOUT_S, or the process died
            Exception: any other error raised by ngspyce or `on_wait`
        """
        try:
            conn = self._child[1] if self._child else self._start()
            conn.send((netlist, vector))
            reply = _receive(conn, self._on_wait)
        except (EOFError, OSError) as e:
            self.close()
            raise SpiceError("ngspice process exited unexpectedly") from e
        except BaseException:
            self.close()
            raise
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, complex):
            raise SpiceError(f"Unexpected reply from ngspice process: {reply!r}")
        return reply

    def close(self) -> None:
        """Kill the child process; a no-op if none is running."""
        if self._child is None:
            return
        process, conn = self._child
        self._child = None
        # Kill before closing the pipe, so the child never sees EOF and reports it.
        process.kill()
        process.join()
        process.close()
        conn.close()

    def _start(self) -> Connection:
        """Start the child and wait until it has loaded libngspice.

        Afterwards the child always waits in recv, so sending never blocks on it.
        """
        # fork would duplicate the GUI toolkit state of the parent
        ctx = multiprocessing.get_context("spawn")
        ctx.set_executable(_python_interpreter())
        conn, child_conn = ctx.Pipe()
        process = ctx.Process(target=_serve, args=(child_conn,), daemon=True)
        try:
            # Only the child may keep its end, so its death reaches `conn` as EOF.
            with child_conn:
                try:
                    process.start()
                except BaseException:
                    conn.close()
                    raise
            self._child = (process, conn)
            loaded = _receive(conn, self._on_wait)
        except (EOFError, OSError, SpiceError) as e:
            raise SpiceUnavailableError(
                f"The ngspice process could not be started ({str(e) or type(e).__name__})."
            ) from e
        if isinstance(loaded, Exception):
            raise loaded
        return conn


def _python_interpreter() -> str:
    """Interpreter for the child process.

    Embedded in KiCad, sys.executable can be the KiCad binary itself; the
    interpreter of that Python installation is then derived from its prefix.
    """
    if os.path.basename(sys.executable).lower().startswith("python"):
        return sys.executable
    interpreter = os.path.join(
        sys.exec_prefix, "python.exe" if os.name == "nt" else "bin/python3"
    )
    if not (os.path.isfile(interpreter) and os.access(interpreter, os.X_OK)):
        raise SpiceUnavailableError(
            f"No Python interpreter found to run ngspice (tried {interpreter}).\n\n"
            'Use the IPC plugin "Parasitics (IPC)" instead.'
        )
    return interpreter


def _receive(conn: Connection, on_wait: Callable[[], None]) -> object:
    """Wait for the child's next reply and re-emit the log records sent with it."""
    deadline = time.monotonic() + SPICE_TIMEOUT_S
    while True:
        on_wait()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SpiceError(
                f"ngspice did not answer within {SPICE_TIMEOUT_S:.0f}s "
                "(likely non-convergent or pathological circuit)"
            )
        if conn.poll(min(_POLL_INTERVAL_S, remaining)):
            break
    records: list[logging.LogRecord]
    reply, records = conn.recv()
    for record in records:
        logging.getLogger(record.name).handle(record)
    return reply


def _serve(conn: Connection) -> None:
    """Child process: report loading libngspice, then answer each request.

    Every reply is the value or the exception, together with the log records
    produced meanwhile for the parent's handlers. Before replying, the circuit
    and its results are unloaded so memory does not grow across requests; if
    that fails the process ends without replying, as its state is undefined.
    """
    records: SimpleQueue[logging.LogRecord] = SimpleQueue()
    logging.getLogger().addHandler(QueueHandler(records))

    def answer(reply: complex | Exception | None) -> None:
        conn.send((reply, [records.get() for _ in range(records.qsize())]))

    try:
        ngspyce = _load_ngspyce()
    except Exception as e:  # noqa: BLE001 - re-raised in the parent
        answer(_transferable(e))
        return
    answer(None)
    while True:
        netlist, vector = conn.recv()
        reply: complex | Exception
        try:
            reply = _measure(ngspyce, netlist, vector)
        except Exception as e:  # noqa: BLE001 - re-raised in the parent
            reply = _transferable(e)
        _unload(ngspyce)
        answer(reply)


def _transferable(error: Exception) -> Exception:
    """`error` if it survives the pipe, else a SpiceError carrying its repr."""
    try:
        pickle.loads(pickle.dumps(error))
    except Exception:  # noqa: BLE001 - any pickling failure
        return SpiceError(repr(error))
    return error


def _load_ngspyce() -> ModuleType:
    """Import ngspyce, which loads libngspice on first import."""
    try:
        try:
            from . import ngspyce
        except ImportError:
            import ngspyce
    except (ImportError, OSError) as e:
        raise SpiceUnavailableError(
            f"The ngspice library could not be loaded ({e}).\n\n"
            "Install ngspice (macOS: brew install libngspice) or set the "
            "LIBNGSPICE environment variable to the library path."
        ) from e
    return ngspyce


def _unload(ngspyce: ModuleType) -> None:
    """Remove all result plots and the circuit, the only one loaded."""
    ngspyce.cmd("destroy all")
    ngspyce.cmd("remcirc")


def _measure(ngspyce: ModuleType, netlist: list[str], vector: str) -> complex:
    plots_before = set(ngspyce.plots())
    if ngspyce.circ(netlist) != 0:
        raise SpiceError("ngspice rejected the netlist")
    ngspyce.cmd("run")
    new_plots = set(ngspyce.plots()) - plots_before
    if len(new_plots) != 1:
        raise SpiceError(f"Expected one new result plot, got {sorted(new_plots)}")
    values = ngspyce.vector(vector, new_plots.pop())
    if len(values) != 1:
        raise SpiceError(f"Expected one value of {vector}, got {values}")
    return complex(values[0])

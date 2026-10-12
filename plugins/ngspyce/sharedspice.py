from __future__ import annotations

import logging
import os
import platform
import sys
from contextlib import ExitStack
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_short,
    c_void_p,
)
from ctypes.util import find_library
from pathlib import Path

logger = logging.getLogger(__name__)


def _kicad_library_dir() -> str | None:
    """Directory of the libngspice that KiCad ships, if this is KiCad's Python.

    The plugin runs on KiCad's bundled Python (IPC venv or pcbnew), which lies
    inside the KiCad installation on macOS and Windows; KiCad loads ngspice from
    Contents/PlugIns/sim of the app bundle and from its bin directory there.
    """
    prefix = os.path.abspath(sys.base_prefix)
    if platform.system() == "Darwin":
        bundle = next((p for p in Path(prefix).parents if p.suffix == ".app"), None)
        return str(bundle / "Contents" / "PlugIns" / "sim") if bundle else None
    if os.name == "nt":
        return prefix  # KiCad's python.exe sits next to its DLLs in bin
    return None


def _library_candidates() -> list[str]:
    """libngspice locations in the order to try; LIBNGSPICE overrides all others."""
    explicit = os.environ.get("LIBNGSPICE")
    if explicit:
        return [explicit]
    if os.name == "nt":
        drive = os.getenv("SystemDrive") or "C:"
        # The default install folder of the official ngspice build
        spice_dir = os.path.join(
            drive, os.sep, "Spice64" if sys.maxsize > 2**32 else "Spice"
        )
        names, fallbacks = (
            ["ngspice.dll", "libngspice-0.dll"],
            [os.path.join(spice_dir, "bin_dll", "ngspice.dll")],
        )
    elif platform.system() == "Darwin":
        names, fallbacks = (
            ["libngspice.0.dylib", "libngspice.dylib"],
            ["/opt/homebrew/lib/libngspice.dylib"],
        )
    else:
        names, fallbacks = [], []
    kicad_dir = _kicad_library_dir()
    bundled = [os.path.join(kicad_dir, name) for name in names] if kicad_dir else []
    system = find_library("ngspice")
    return ([system] if system else []) + bundled + fallbacks


def _use_installation_scripts(library: str) -> None:
    """Let ngspice read spinit from the installation the library belongs to.

    Unset, SPICE_LIB_DIR falls back to a location compiled into ngspice, which
    need not exist for a library loaded from another installation.
    """
    lib_dir = os.path.join(
        os.path.dirname(os.path.dirname(library)), "share", "ngspice"
    )
    spinit = os.path.join(lib_dir, "scripts", "spinit")
    if os.path.isabs(library) and os.path.isfile(spinit):
        os.environ.setdefault("SPICE_LIB_DIR", lib_dir)


# Windows DLL directories of the loaded library, registered for the process
# so that ngspice also finds the libraries it loads later.
_dll_directories: list[ExitStack] = []


def _load_library() -> CDLL:
    tried = []
    for candidate in _library_candidates():
        if os.path.isabs(candidate) and not os.path.isfile(candidate):
            tried.append(f"{candidate} (missing)")
            continue
        with ExitStack() as dll_directory:
            if os.name == "nt" and os.path.isabs(candidate):
                dll_directory.enter_context(
                    os.add_dll_directory(os.path.dirname(candidate))
                )
            try:
                library = CDLL(candidate)
            except OSError as e:
                tried.append(f"{candidate} ({e})")
                continue
            _dll_directories.append(dll_directory.pop_all())
            _use_installation_scripts(candidate)
            return library
    raise OSError(f"ngspice library not found (tried: {'; '.join(tried) or 'nothing'})")


spice = _load_library()

# libngspice source code is listed before the relevant ctype structs

captured_output = []


@CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
def printfcn(output: bytes, _id: int, _ret: int) -> int:
    """Callback for libngspice to print a message"""
    prefix, _, content = output.decode("ascii").partition(" ")
    if prefix == "stderr":
        logger.error(content)
    else:
        captured_output.append(content)
    return 0


@CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
def statfcn(status: bytes, _id: int, _ret: int) -> int:
    """
    Callback for libngspice to report simulation status like 'tran 5%'
    """
    logger.debug(status.decode("ascii"))
    return 0


@CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)
def controlled_exit(
    exit_status: int,
    immediate_unloading: bool,
    requested_exit: bool,
    libngspice_id: int,
    ret: int,
) -> int:
    logger.debug(
        "ControlledExit: %s",
        {
            "exit_status": exit_status,
            "immediate_unloading": immediate_unloading,
            "requested_exit": requested_exit,
            "libngspice_id": libngspice_id,
            "ret": ret,
        },
    )
    return 0


# typedef struct vecvalues {
# char* name; /* name of a specific vector */
# double creal; /* actual data value */
# double cimag; /* actual data value */
# bool is_scale;/* if 'name' is the scale vector */
# bool is_complex;/* if the data are complex numbers */
# } vecvalues, *pvecvalues;


class vecvalues(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("creal", c_double),
        ("cimag", c_double),
        ("is_scale", c_bool),
        ("is_complex", c_bool),
    ]


# typedef struct vecvaluesall {
# int veccount; /* number of vectors in plot */
# int vecindex; /* index of actual set of vectors. i.e. the number of accepted data points */
# pvecvalues *vecsa; /* values of actual set of vectors, indexed from 0 to veccount - 1 */
# } vecvaluesall, *pvecvaluesall;


class vecvaluesall(Structure):
    _fields_ = [
        ("veccount", c_int),
        ("vecindex", c_int),
        ("vecsa", POINTER(POINTER(vecvalues))),
    ]


@CFUNCTYPE(c_int, POINTER(vecvaluesall), c_int, c_int, c_void_p)
def send_data(
    vecvaluesall_: vecvaluesall, num_structs: int, libngspice_id: int, ret: int
) -> int:
    logger.debug(
        "SendData: %s",
        {
            "vecvaluesall": vecvaluesall_,
            "num_structs": num_structs,
            "libngspice_id": libngspice_id,
            "ret": ret,
        },
    )
    return 0


# int  ngSpice_Command(char* command);
spice.ngSpice_Command.argtypes = [c_char_p]

# int ngSpice_Circ(char**)
spice.ngSpice_Circ.argtypes = [POINTER(c_char_p)]
spice.ngSpice_AllPlots.restype = POINTER(c_char_p)

spice.ngSpice_AllVecs.argtypes = [c_char_p]
spice.ngSpice_AllVecs.restype = POINTER(c_char_p)
spice.ngSpice_CurPlot.restype = c_char_p


# struct ngcomplex {
#    double cx_real;
#    double cx_imag;
# } ;


class ngcomplex(Structure):
    _fields_ = [("cx_real", c_double), ("cx_imag", c_double)]


# /* Dvec flags. */
# enum dvec_flags {
#   VF_REAL = (1 << 0),       /* The data is real. */
#   VF_COMPLEX = (1 << 1),    /* The data is complex. */
#   VF_ACCUM = (1 << 2),      /* writedata should save this vector. */
#   VF_PLOT = (1 << 3),       /* writedata should incrementally plot it. */
#   VF_PRINT = (1 << 4),      /* writedata should print this vector. */
#   VF_MINGIVEN = (1 << 5),   /* The v_minsignal value is valid. */
#   VF_MAXGIVEN = (1 << 6),   /* The v_maxsignal value is valid. */
#   VF_PERMANENT = (1 << 7)   /* Don't garbage collect this vector. */
# };


class dvec_flags:
    vf_real = 1 << 0  # The data is real.
    vf_complex = 1 << 1  # The data is complex.
    vf_accum = 1 << 2  # writedata should save this vector.
    vf_plot = 1 << 3  # writedata should incrementally plot it.
    vf_print = 1 << 4  # writedata should print this vector.
    vf_mingiven = 1 << 5  # The v_minsignal value is valid.
    vf_maxgiven = 1 << 6  # The v_maxsignal value is valid.
    vf_permanent = 1 << 7  # Don't garbage collect this vector.


# /* vector info obtained from any vector in ngspice.dll.
# Allows direct access to the ngspice internal vector structure,
# as defined in include/ngspice/devc.h .*/
# typedef struct vector_info {
#    char *v_name;		/* Same as so_vname. */
#    int v_type;			/* Same as so_vtype. */
#    short v_flags;		/* Flags (a combination of VF_*). */
#    double *v_realdata;		/* Real data. */
#    ngcomplex_t *v_compdata;	/* Complex data. */
#    int v_length;		/* Length of the vector. */
# } vector_info, *pvector_info;


class vector_info(Structure):
    _fields_ = [
        ("v_name", c_char_p),
        ("v_type", c_int),
        ("v_flags", c_short),
        ("v_realdata", POINTER(c_double)),
        ("v_compdata", POINTER(ngcomplex)),
        ("v_length", c_int),
    ]


# /* get info about a vector */
# pvector_info ngGet_Vec_Info(char* vecname);
spice.ngGet_Vec_Info.restype = POINTER(vector_info)
spice.ngGet_Vec_Info.argtypes = [c_char_p]

# Unit names for use with pint or other unit libraries
vector_type = [
    "dimensionless",  # notype = 0
    "second",  # time = 1
    "hertz",  # frequency = 2
    "volt",  # voltage = 3
    "ampere",  # current = 4
    "NotImplemented",  # output_n_dens = 5
    "NotImplemented",  # output_noise = 6
    "NotImplemented",  # input_n_dens = 7
    "NotImplemented",  # input_noise = 8
    "NotImplemented",  # pole = 9
    "NotImplemented",  # zero = 10
    "NotImplemented",  # sparam = 11
    "NotImplemented",  # temp = 12
    "ohm",  # res = 13
    "ohm",  # impedance = 14
    "siemens",  # admittance = 15
    "watt",  # power = 16
    "dimensionless",  # phase = 17
    "NotImplemented",  # db = 18
    "farad",  # capacitance = 19
    "coulomb",  # charge = 21
]


#
# enum simulation_types {
#   ...
# };
class simulation_type:
    notype = 0
    time = 1
    frequency = 2
    voltage = 3
    current = 4
    output_n_dens = 5
    output_noise = 6
    input_n_dens = 7
    input_noise = 8
    pole = 9
    zero = 10
    sparam = 11
    temp = 12
    res = 13
    impedance = 14
    admittance = 15
    power = 16
    phase = 17
    db = 18
    capacitance = 19
    charge = 20

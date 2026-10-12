import multiprocessing

# A spawned ngspice process imports this package only to reach spice_worker;
# its process name is already set while it does so.
if multiprocessing.current_process().name == "MainProcess":
    from . import pcbnew_plugin  # noqa: F401 - registers the action plugin

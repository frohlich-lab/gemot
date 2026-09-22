"""The harness's non-LLM model producer: a neural ODE fitted to a PEtab problem's data.

:mod:`.problem` reads what the problem's tables say about its states, knobs and starts without
opening the shipped model; :mod:`.stage` writes that as a PEtab SciML neural-ODE problem; and
:mod:`.train` fits it through AMICI's JAX backend. Nothing here is imported by the server or the
panel, and this package imports none of its modules eagerly: each pulls in petab, AMICI or JAX.
"""

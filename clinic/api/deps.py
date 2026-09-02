"""The single dependency: one process-wide ``ClinicStore`` (the in-memory session), injected into the
routes via ``get_store``."""
from clinic.core.store import ClinicStore

_store = ClinicStore()


def get_store() -> ClinicStore:
    return _store

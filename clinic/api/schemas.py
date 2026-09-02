"""Pydantic request bodies for the clinic API."""
from typing import Literal

from pydantic import BaseModel


class OpenIn(BaseModel):
    """Step 1 of upload: decode an E2E and list its 6x6-measurable scans (no GA yet)."""
    path: str


class ProcessIn(BaseModel):
    """Step 2 of upload: process the chosen scan with the chosen Bruch's-membrane source."""
    path: str
    index: int
    bm_choice: Literal["device", "dl", "auto"]


class ReopenIn(BaseModel):
    """Re-open a scan already recorded in the patient database, by its record id."""
    record_id: str

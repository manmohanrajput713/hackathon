"""Retrieve the Abdalla et al. (2022) PDF and the PGEHackathon 2023 ESP files.

The hackathon repository is https://github.com/PGEHackathon/data2023.
PMC serves the open-access PDF behind a short SHA-256 proof-of-work cookie.
``curl`` is used for the transfer because the system Python trust store on
this workstation rejects the publisher certificate chain.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PAPER_DOI = "10.1021/acsomega.1c05881"
PMC_PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9161246/pdf/ao1c05881.pdf"
CLAUSTHAL_PDF_URL = (
    "https://dokumente.ub.tu-clausthal.de/servlets/MCRFileNodeServlet/"
    "clausthal_derivate_00001889/acsomega.1c05881.pdf"
)
HACKATHON_RAW = "https://raw.githubusercontent.com/PGEHackathon/data2023/main"
DATA_FILES = {
    "dailyData.csv": f"{HACKATHON_RAW}/dailyData.csv",
    "wellData.csv": f"{HACKATHON_RAW}/wellData.csv",
    "solution.csv": f"{HACKATHON_RAW}/solution.csv",
}

DAILY_REQUIRED = [
    "OIL",
    "GAS",
    "WATER",
    "ARTIFICIAL_LIFT",
    "DOWN_TIME_HOURS",
    "ESP Data - Drive Current",
    "ESP Data - Drive Voltage",
    "ESP Data - Intake Pressure",
    "ESP Data - Motor Winding Temperature",
    "ESP Data - Output Frequency",
    "Discharge_Pressure",
    "Pump_Average_Pressure",
    "ESP_Temperature",
    "AL_Key",
    "Well_ID",
]
WELL_REQUIRED = [
    "Well_ID",
    "AL_Key",
    "Artificial_Lift_Type",
    "AL_Bottom_Depth",
    "ESP_Pump_Stages",
    "DLS_Critical",
    "ESP_Motor_Frequency_Rating",
    "ESP_Motor_Current_Rating",
    "ESP_Motor_Voltage_Rating",
    "ESP_Motor_Power_Rating",
    "DLS_at_Set_Depth",
]
SOLUTION_REQUIRED = ["Well_ID", "AL_Key", "Fail in 30 days"]
SOLUTION_LABEL = "Fail in 30 days"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class SchemaError(ValueError):
    """Raised when a downloaded table is missing columns the pipeline uses."""


def _curl(url: str, dest: Path, cookie_jar: Path | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-fsSL",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--max-time",
        "180",
        "-A",
        USER_AGENT,
        "-o",
        str(dest),
        url,
    ]
    if cookie_jar is not None:
        command[1:1] = ["-b", str(cookie_jar), "-c", str(cookie_jar)]
    subprocess.run(command, check=True)


def _is_pdf(path: Path, min_bytes: int = 1_000_000) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    return path.read_bytes()[:5] == b"%PDF-"


def _solve_pmc_proof_of_work(html: str) -> tuple[str, str]:
    """Return ``(cookie_name, cookie_value)`` for a PMC viewer challenge page.

    The viewer hashes ``challenge + nonce`` with SHA-256 until the hex digest
    starts with ``difficulty`` zeros, then stores ``challenge,nonce``.
    """
    challenge = re.search(r'POW_CHALLENGE = "([^"]+)"', html)
    difficulty_match = re.search(r'POW_DIFFICULTY = "(\d+)"', html)
    cookie_match = re.search(r'POW_COOKIE_NAME = "([^"]+)"', html)
    if not (challenge and difficulty_match and cookie_match):
        raise RuntimeError("PMC proof-of-work page did not contain a challenge.")
    difficulty = int(difficulty_match.group(1))
    prefix = "0" * difficulty
    nonce = 0
    text = challenge.group(1)
    while True:
        digest = hashlib.sha256(f"{text}{nonce}".encode()).hexdigest()
        if digest.startswith(prefix):
            logger.info("PMC proof-of-work solved at nonce %s.", nonce)
            return cookie_match.group(1), f"{text},{nonce}"
        nonce += 1


def _write_cookie(path: Path, name: str, value: str) -> None:
    # A leading-dot domain sends the cookie to www.ncbi.nlm.nih.gov as well.
    line = f".ncbi.nlm.nih.gov\tTRUE\t/\tTRUE\t1999999999\t{name}\t{value}\n"
    path.write_text("# Netscape HTTP Cookie File\n" + line, encoding="utf-8")


def download_paper(dest: Path, force: bool = False) -> Path:
    """Save the open-access article to ``dest`` when the network allows it."""
    dest = Path(dest)
    if dest.exists() and not force and _is_pdf(dest):
        logger.info("Using existing paper PDF at %s.", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    scratch = dest.with_suffix(".download")
    try:
        _curl(PMC_PDF_URL, scratch)
    except subprocess.CalledProcessError as exc:
        logger.warning("Direct PMC download failed (%s). Trying the repository mirror.", exc)
        _curl(CLAUSTHAL_PDF_URL, scratch)
    payload = scratch.read_bytes()
    if payload[:5] != b"%PDF-" and b"POW_CHALLENGE" in payload:
        cookie_name, cookie_value = _solve_pmc_proof_of_work(payload.decode("utf-8", "replace"))
        jar = scratch.with_suffix(".cookies")
        _write_cookie(jar, cookie_name, cookie_value)
        _curl(PMC_PDF_URL, scratch, cookie_jar=jar)
        jar.unlink(missing_ok=True)
    if not _is_pdf(scratch, min_bytes=100_000):
        scratch.unlink(missing_ok=True)
        raise RuntimeError(
            "The paper PDF could not be downloaded. Place the open-access file at "
            f"{dest} and rerun with --skip-download."
        )
    scratch.replace(dest)
    logger.info("Saved paper PDF (%s bytes) to %s.", dest.stat().st_size, dest)
    return dest


def download_hackathon(dest_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download ``dailyData.csv``, ``wellData.csv``, and ``solution.csv``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, url in DATA_FILES.items():
        target = dest_dir / name
        if target.exists() and not force and target.stat().st_size > 0:
            logger.info("Using existing %s.", target)
        else:
            logger.info("Downloading %s.", url)
            _curl(url, target)
        saved[name] = target
    return saved


def schema_check(daily, well, solution) -> dict:
    """Validate the columns and keys the KDD pipeline relies on.

    The hackathon README describes ``Status`` and ``Failure_Type``. The files
    published on ``main`` do not contain those fields. Failure membership is
    taken from the README rule that ESPs absent from ``solution.csv`` have
    failed, together with the published ``Fail in 30 days`` column.
    """
    missing = {
        "dailyData.csv": [c for c in DAILY_REQUIRED if c not in daily.columns],
        "wellData.csv": [c for c in WELL_REQUIRED if c not in well.columns],
        "solution.csv": [c for c in SOLUTION_REQUIRED if c not in solution.columns],
    }
    problems = {name: cols for name, cols in missing.items() if cols}
    if problems:
        raise SchemaError(f"Required columns are absent: {problems}")

    well_keys = well[["Well_ID", "AL_Key"]].copy()
    if well_keys.duplicated().any():
        raise SchemaError("wellData.csv has duplicate (Well_ID, AL_Key) rows.")
    solution_keys = solution[["Well_ID", "AL_Key"]].copy()
    if solution_keys.duplicated().any():
        raise SchemaError("solution.csv has duplicate (Well_ID, AL_Key) rows.")
    labels = set(solution[SOLUTION_LABEL].dropna().unique())
    if not labels <= {0, 1}:
        raise SchemaError(f"{SOLUTION_LABEL} contains values outside {{0, 1}}: {labels}")
    if len(daily) < 1000:
        raise SchemaError(f"dailyData.csv has too few rows ({len(daily)}).")

    advisories = []
    for column in ("Status", "Failure_Type", "Failure_Type_Detail"):
        if column not in well.columns:
            advisories.append(
                f"wellData.csv has no {column} column. "
                "Labels use solution membership instead of a workover sheet."
            )
    if "Date" not in daily.columns and "DATE" not in daily.columns:
        advisories.append(
            "dailyData.csv has no calendar timestamp. "
            "Row order within each ESP is the daily time index."
        )
    report = {
        "daily_rows": int(len(daily)),
        "daily_columns": int(daily.shape[1]),
        "well_rows": int(len(well)),
        "solution_rows": int(len(solution)),
        "solution_fail30_counts": {
            str(k): int(v) for k, v in solution[SOLUTION_LABEL].value_counts().items()
        },
        "advisories": advisories,
    }
    for note in advisories:
        logger.warning(note)
    logger.info(
        "Schema check passed: %s daily rows, %s ESPs, %s solution rows.",
        report["daily_rows"],
        report["well_rows"],
        report["solution_rows"],
    )
    return report

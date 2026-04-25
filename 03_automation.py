"""
============================================================
 Smart Sales Forecasting System
 Module 3: Automation Engine
============================================================
 Features:
   • Auto-detect new data files
   • Run full pipeline (clean → predict → report)
   • Email report dispatch simulation
   • Scheduled runs (cron-style)
   • Audit logging
============================================================
"""

import os
import time
import hashlib
import logging
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path

# ── Setup logging ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("automation.log"),
    ]
)
log = logging.getLogger("SalesAutomation")

# ── Colors ────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

AUDIT_LOG = "audit_trail.json"
WATCH_DIR  = Path(".")
STATE_FILE = "watcher_state.json"


# ════════════════════════════════════════════════════════
#  Utility: File fingerprinting (change detection)
# ════════════════════════════════════════════════════════

def file_hash(path: str) -> str:
    """Return MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ════════════════════════════════════════════════════════
#  Audit Trail
# ════════════════════════════════════════════════════════

def audit(event: str, details: dict = {}):
    """Append an event to the JSON audit log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **details,
    }
    log_data = []
    if os.path.exists(AUDIT_LOG):
        with open(AUDIT_LOG) as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []
    log_data.append(entry)
    with open(AUDIT_LOG, "w") as f:
        json.dump(log_data, f, indent=2)


# ════════════════════════════════════════════════════════
#  Pipeline Orchestrator
# ════════════════════════════════════════════════════════

def run_pipeline(source_file: str) -> bool:
    """Run the full clean → predict → report pipeline."""
    start = datetime.now()
    log.info(f"🚀  Pipeline triggered by: {source_file}")
    audit("PIPELINE_START", {"source": source_file})

    steps = [
        ("Data Cleaning",    "python 01_data_cleaning.py"),
        ("AI Prediction",    "python 02_ai_prediction.py"),
        ("Report Generation","python 03_report_generator.py"),
    ]

    for step_name, cmd in steps:
        log.info(f"   ▶  Running: {step_name}...")
        ret = os.system(cmd)
        if ret != 0:
            log.error(f"   ✗  Step failed: {step_name}")
            audit("STEP_FAILED", {"step": step_name, "cmd": cmd})
            return False
        log.info(f"   ✓  {step_name} complete")
        audit("STEP_COMPLETE", {"step": step_name})

    elapsed = (datetime.now() - start).seconds
    log.info(f"✅  Full pipeline finished in {elapsed}s")
    audit("PIPELINE_COMPLETE", {"elapsed_seconds": elapsed})
    return True


# ════════════════════════════════════════════════════════
#  Email Dispatcher (Simulated)
# ════════════════════════════════════════════════════════

def dispatch_report(recipients: list, report_path: str, forecast_summary: dict):
    """Simulate sending email report (replace with SMTP in production)."""
    log.info(f"📧  Dispatching report to {len(recipients)} recipients...")

    email_body = f"""
Subject: 📊 Automated Sales Forecast Report — {datetime.now().strftime('%B %Y')}

Hello Team,

Your automated Sales Forecast Report is ready.

── QUICK SUMMARY ──────────────────────────────
  Next Month Forecast  : ${forecast_summary.get('next_month', 0):,.0f}
  3-Month Total        : ${forecast_summary.get('total_3m', 0):,.0f}
  Expected Growth      : {forecast_summary.get('growth_pct', 0):.1f}%
  Model Accuracy (R²)  : {forecast_summary.get('r2', 0):.3f}
────────────────────────────────────────────────

📎 Full report attached: {report_path}

This report was generated automatically by the Smart Sales
Forecasting System. Next run scheduled for:
{(datetime.now() + timedelta(days=7)).strftime('%A, %B %d at 08:00 AM')}

──────────────────────────────────────────────
Smart Sales Forecasting System | Auto-Reporting Engine
"""

    for recipient in recipients:
        print(f"   📤  [SIMULATED] Email → {recipient}")
        print(f"   {'-'*52}")
        # In production: smtplib.SMTP(...) here
        time.sleep(0.3)

    log.info(f"✅  Report dispatched (simulated mode)")
    audit("EMAIL_DISPATCHED", {
        "recipients": recipients,
        "report": report_path,
        "simulated": True
    })


# ════════════════════════════════════════════════════════
#  File Watcher (Polling-based)
# ════════════════════════════════════════════════════════

def watch_for_new_data(watch_pattern="*.csv", interval_sec=10, max_cycles=3):
    """Poll directory for new/changed CSV files and trigger pipeline."""
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║   👁  File Watcher Active                           ║
║   Watching: {str(WATCH_DIR):<39}║
║   Pattern : {watch_pattern:<39}║
╚══════════════════════════════════════════════════════╝
{RESET}""")

    state = load_state()
    cycles = 0

    while cycles < max_cycles:
        cycles += 1
        log.info(f"[Cycle {cycles}/{max_cycles}] Scanning for changes...")

        changed_files = []
        for fpath in WATCH_DIR.glob(watch_pattern):
            fstr = str(fpath)
            h    = file_hash(fstr)
            if fstr not in state or state[fstr] != h:
                changed_files.append(fstr)
                state[fstr] = h
                log.info(f"  📄  Change detected: {fpath.name}")

        if changed_files:
            save_state(state)
            for f in changed_files:
                if "raw" in f.lower():
                    log.info(f"  🔥  New raw data! Running pipeline for: {f}")
                    run_pipeline(f)
                    dispatch_report(
                        recipients=["sales@company.com",
                                    "director@company.com",
                                    "data-team@company.com"],
                        report_path="sales_report.pdf",
                        forecast_summary={
                            "next_month": 185000,
                            "total_3m":   560000,
                            "growth_pct": 8.3,
                            "r2":         0.921,
                        }
                    )
        else:
            log.info("  ✓  No changes detected.")

        if cycles < max_cycles:
            log.info(f"  ⏱  Next scan in {interval_sec}s... (Ctrl+C to stop)")
            time.sleep(interval_sec)

    log.info("🏁  Watcher completed all cycles.")


# ════════════════════════════════════════════════════════
#  Scheduler: Weekly Cron-style runner
# ════════════════════════════════════════════════════════

class ScheduledRunner:
    """Simple weekly scheduler simulation."""

    SCHEDULE = {
        "weekly_report": {"day": "Monday", "hour": 8},
        "monthly_deep":  {"day": "1st of month", "hour": 7},
    }

    def print_schedule(self):
        print(f"\n{YELLOW}{BOLD}📅  Automation Schedule:{RESET}")
        for job, cfg in self.SCHEDULE.items():
            print(f"   • {job:<20} → {cfg['day']} at {cfg['hour']:02d}:00")

    def simulate_weekly(self):
        """Simulate a scheduled weekly run."""
        log.info("⏰  Scheduled weekly run triggered")
        audit("SCHEDULED_RUN", {"type": "weekly"})
        now = datetime.now()
        print(f"\n{GREEN}  ⏰  Weekly Run: {now.strftime('%A %Y-%m-%d %H:%M')}{RESET}")
        run_pipeline("sales_data_raw.csv")


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║   ⚡  Sales Automation Engine  v1.0                 ║
║   Auto-Detect → Pipeline → Email → Schedule         ║
╚══════════════════════════════════════════════════════╝
{RESET}""")

    scheduler = ScheduledRunner()
    scheduler.print_schedule()

    print(f"\n{YELLOW}Choose mode:{RESET}")
    print("  1 → Run full pipeline now")
    print("  2 → Start file watcher (demo: 3 cycles)")
    print("  3 → Simulate scheduled weekly run")
    print("  4 → Show audit trail")

    choice = input("\n  Enter choice [1-4]: ").strip()

    if choice == "1":
        run_pipeline("sales_data_raw.csv")
        dispatch_report(
            ["sales@company.com", "director@company.com"],
            "sales_report.pdf",
            {"next_month": 185000, "total_3m": 560000,
             "growth_pct": 8.3, "r2": 0.921}
        )
    elif choice == "2":
        watch_for_new_data(interval_sec=5, max_cycles=3)
    elif choice == "3":
        scheduler.simulate_weekly()
    elif choice == "4":
        if os.path.exists(AUDIT_LOG):
            with open(AUDIT_LOG) as f:
                events = json.load(f)
            print(f"\n{CYAN}Audit Trail ({len(events)} events):{RESET}")
            for e in events[-10:]:
                print(f"  {e['timestamp']}  |  {e['event']}")
        else:
            print("  No audit trail yet.")
    else:
        print("Running default pipeline...")
        run_pipeline("sales_data_raw.csv")

if __name__ == "__main__":
    main()

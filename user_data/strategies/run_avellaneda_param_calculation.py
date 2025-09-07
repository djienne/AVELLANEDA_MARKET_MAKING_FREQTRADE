#!/usr/bin/env python3
"""
Avellaneda parameter calculation runner with duplicate execution protection and comprehensive logging.
This script prevents running the calculation more than once within a 24-hour period.
Compatible with both Windows and Linux systems.
"""

import os
import sys
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path


def setup_logging():
    """
    Set up logging configuration to write to both file and console.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Define log file path
    current_dir = Path(__file__).parent.absolute()
    log_file_path = current_dir / "avellaneda_runner.log"
    
    # Create logger
    logger = logging.getLogger('AvellanedaRunner')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file_path


def log_separator(logger, title, char='=', width=80):
    """Log a separator line with title."""
    logger.info(char * width)
    logger.info(title.center(width))
    logger.info(char * width)


def log_subprocess_output(logger, stdout, stderr, title_prefix="SUBPROCESS"):
    """Log subprocess output with proper formatting."""
    if stdout:
        logger.info(f"{'-' * 80}")
        logger.info(f"{title_prefix} STDOUT:")
        logger.info(f"{'-' * 80}")
        for line in stdout.strip().split('\n'):
            if line.strip():  # Only log non-empty lines
                logger.info(f"STDOUT: {line}")
    
    if stderr:
        logger.info(f"{'-' * 80}")
        logger.info(f"{title_prefix} STDERR:")
        logger.info(f"{'-' * 80}")
        for line in stderr.strip().split('\n'):
            if line.strip():  # Only log non-empty lines
                logger.info(f"STDERR: {line}")


def run_avellaneda_param_calculation():
    """
    Executes the Avellaneda parameter calculation script with duplicate run protection.
    
    Returns:
        dict: Result dictionary with status and message
    """
    
    # Set up logging
    logger, log_file_path = setup_logging()
    
    # Log execution start
    log_separator(logger, "AVELLANEDA PARAMETER CALCULATION RUNNER - START")
    logger.info(f"Execution started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Log file location: {log_file_path}")
    
    # Define paths (works on both Windows and Linux)
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.parent  # Go up two levels to reach project root
    script_path = project_root / "scripts" / "calculate_avellaneda_parameters.py"
    lock_file_path = current_dir / ".avellaneda_last_run.json"
    
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Lock file path: {lock_file_path}")
    
    # Check if the calculation script exists
    if not script_path.exists():
        error_msg = f"Calculation script not found at: {script_path}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }
    
    logger.info(f"Calculation script found: {script_path}")
    
    # Check for previous execution within 24 hours or if it's a new day
    if lock_file_path.exists():
        logger.info(f"Lock file exists, checking previous execution time...")
        try:
            with open(lock_file_path, 'r') as f:
                lock_data = json.load(f)
            
            last_run_time = datetime.fromisoformat(lock_data.get('last_run', '1970-01-01T00:00:00'))
            # Convert to UTC if not already timezone-aware
            if last_run_time.tzinfo is None:
                last_run_time = last_run_time.replace(tzinfo=timezone.utc)
            
            current_time_utc = datetime.now(timezone.utc)
            last_run_time_utc = last_run_time.astimezone(timezone.utc)
            
            logger.info(f"Last run time (UTC): {last_run_time_utc.isoformat()}")
            logger.info(f"Current time (UTC): {current_time_utc.isoformat()}")
            
            # Check if it's a new UTC day (with 5-minute grace period after midnight)
            current_date_utc = current_time_utc.date()
            last_run_date_utc = last_run_time_utc.date()
            
            # Calculate minutes since midnight UTC
            midnight_utc_today = datetime.combine(current_date_utc, datetime.min.time()).replace(tzinfo=timezone.utc)
            minutes_since_midnight = (current_time_utc - midnight_utc_today).total_seconds() / 60
            
            logger.info(f"Current UTC date: {current_date_utc}")
            logger.info(f"Last run UTC date: {last_run_date_utc}")
            logger.info(f"Minutes since midnight UTC: {minutes_since_midnight:.2f}")
            
            # Allow rerun if it's a new day and we're past the 5-minute grace period
            # Also allow at UTC+3 hours (03:00 UTC) to account for data reset timing
            is_new_day_with_grace = (current_date_utc > last_run_date_utc and minutes_since_midnight >= 5)
            is_new_day_at_3am = (current_date_utc > last_run_date_utc and minutes_since_midnight >= 180)  # 3 hours = 180 minutes
            
            time_since_last_run = current_time_utc - last_run_time_utc
            
            logger.info(f"Is new day with grace period (>5min): {is_new_day_with_grace}")
            logger.info(f"Is new day at 3AM (>180min): {is_new_day_at_3am}")
            logger.info(f"Time since last run: {time_since_last_run}")
            
            # Skip only if same day and less than 24 hours have passed, and not at the 3 AM reset time
            if not (is_new_day_with_grace or is_new_day_at_3am) and time_since_last_run < timedelta(hours=24):
                time_remaining = timedelta(hours=24) - time_since_last_run
                hours_remaining = int(time_remaining.total_seconds() // 3600)
                minutes_remaining = int((time_remaining.total_seconds() % 3600) // 60)
                
                # If it's a new day but within grace periods, mention that
                grace_message = ""
                if current_date_utc > last_run_date_utc:
                    if minutes_since_midnight < 5:
                        grace_message = f" (New day detected, but waiting for 5-minute grace period. {5 - int(minutes_since_midnight)} minutes remaining)"
                    elif minutes_since_midnight < 180:
                        grace_message = f" (New day detected, next opportunity at 03:00 UTC. {180 - int(minutes_since_midnight)} minutes remaining)"
                
                skip_message = (f"Calculation already performed within the last 24 hours. "
                              f"Time remaining: {hours_remaining}h {minutes_remaining}m{grace_message}. "
                              f"Will also run on new UTC day after 5-minute grace period or at 03:00 UTC.")
                
                logger.info(f"EXECUTION SKIPPED: {skip_message}")
                
                return {
                    "status": "skipped",
                    "message": skip_message,
                    "last_run": last_run_time.isoformat(),
                    "current_date_utc": current_date_utc.isoformat(),
                    "last_run_date_utc": last_run_date_utc.isoformat(),
                    "minutes_since_midnight_utc": round(minutes_since_midnight, 2)
                }
            elif is_new_day_with_grace:
                logger.info(f"New UTC day detected ({current_date_utc}) and past 5-minute grace period. Proceeding with calculation.")
            elif is_new_day_at_3am:
                logger.info(f"New UTC day detected ({current_date_utc}) and past 03:00 UTC. Proceeding with calculation for data reset.")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            warning_msg = f"Could not read lock file, proceeding with calculation: {e}"
            logger.warning(warning_msg)
    else:
        logger.info("No lock file found, proceeding with first execution.")
    
    # Execute the calculation script
    try:
        logger.info(f"Starting Avellaneda parameter calculation...")
        logger.info(f"Script path: {script_path}")
        
        # Change to the script directory to ensure relative paths work correctly
        original_cwd = os.getcwd()
        script_dir = script_path.parent
        logger.info(f"Changing working directory from {original_cwd} to {script_dir}")
        os.chdir(script_dir)
        
        try:
            # Run the calculation script using Python
            logger.info(f"Executing command: {sys.executable} {script_path}")
            logger.info(f"Timeout set to: 3600 seconds")
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            logger.info(f"Subprocess completed with return code: {result.returncode}")
            
            # Log subprocess output regardless of success/failure
            log_subprocess_output(logger, result.stdout, result.stderr, "CALCULATION SCRIPT")
            
            if result.returncode == 0:
                # Update the lock file with current timestamp in UTC
                current_time = datetime.now(timezone.utc)
                lock_data = {
                    "last_run": current_time.isoformat(),
                    "script_path": str(script_path),
                    "execution_status": "success"
                }
                
                logger.info(f"Updating lock file: {lock_file_path}")
                with open(lock_file_path, 'w') as f:
                    json.dump(lock_data, f, indent=2)
                
                success_message = "Avellaneda parameter calculation completed successfully"
                logger.info(f"SUCCESS: {success_message}")
                
                return {
                    "status": "success",
                    "message": success_message,
                    "execution_time": current_time.isoformat(),
                    "stdout": result.stdout,
                    "stderr": result.stderr if result.stderr else None
                }
            else:
                error_message = f"Calculation script failed with return code {result.returncode}"
                logger.error(f"FAILURE: {error_message}")
                
                return {
                    "status": "error",
                    "message": error_message,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        finally:
            # Always restore the original working directory
            logger.info(f"Restoring working directory to: {original_cwd}")
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        timeout_message = "Calculation script timed out after 2000 seconds"
        logger.error(f"TIMEOUT: {timeout_message}")
        return {
            "status": "error",
            "message": timeout_message
        }
    except Exception as e:
        exception_message = f"Error executing calculation script: {str(e)}"
        logger.error(f"EXCEPTION: {exception_message}")
        return {
            "status": "error",
            "message": exception_message
        }


if __name__ == "__main__":
    """
    Run the function directly when script is executed
    """
    result = run_avellaneda_param_calculation()
    
    # Set up logging for final output (reuse the same logger)
    logger, log_file_path = setup_logging()
    
    log_separator(logger, "EXECUTION SUMMARY")
    logger.info(f"Status: {result['status'].upper()}")
    logger.info(f"Message: {result['message']}")
    
    if result.get('execution_time'):
        logger.info(f"Execution Time: {result['execution_time']}")
    
    if result.get('last_run'):
        logger.info(f"Last Run: {result['last_run']}")
    
    if result.get('current_date_utc'):
        logger.info(f"Current UTC Date: {result['current_date_utc']}")
        logger.info(f"Last Run UTC Date: {result.get('last_run_date_utc', 'N/A')}")
        logger.info(f"Minutes Since Midnight UTC: {result.get('minutes_since_midnight_utc', 'N/A')}")
    
    logger.info(f"Execution Schedule:")
    logger.info(f"  • Daily at 00:05 UTC (5-minute grace period after midnight)")
    logger.info(f"  • Daily at 03:00 UTC (data reset consideration, just in case)")
    logger.info(f"  • After 24 hours from last run (fallback)")
    
    # Final subprocess output logging (if not already logged)
    if result.get('stdout') or result.get('stderr'):
        log_subprocess_output(logger, result.get('stdout'), result.get('stderr'), "FINAL OUTPUT")
    
    log_separator(logger, "EXECUTION COMPLETE")
    logger.info(f"Log file saved to: {log_file_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] in ['success', 'skipped'] else 1)
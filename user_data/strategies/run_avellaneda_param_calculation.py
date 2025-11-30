#!/usr/bin/env python3
"""
Avellaneda parameter calculation runner with duplicate execution protection and comprehensive logging.
This script prevents running the calculation more than once within a configurable window (default: 15 minutes) and
targets the active trading symbol rather than a hard-coded default.
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


def _extract_symbol(raw_symbol):
    """
    Normalize a raw pair/symbol string to a base symbol (e.g., 'PAXG/USDC:USDC' -> 'PAXG').
    Accepts comma-delimited lists and returns the first entry when present.
    """
    if not raw_symbol:
        return None

    symbol = raw_symbol.strip()
    if ',' in symbol:
        symbol = symbol.split(',')[0].strip()
    if '/' in symbol:
        symbol = symbol.split('/')[0].strip()
    if ':' in symbol:
        symbol = symbol.split(':')[0].strip()

    return symbol.upper() if symbol else None


def _determine_trading_symbol(project_root: Path, logger: logging.Logger) -> str:
    """
    Resolve the trading symbol from environment variables or config.
    Priority:
      1) TRADING_SYMBOL / SYMBOL environment variables
      2) user_data/config.json exchange.pair_whitelist[0]
      3) SYMBOLS environment variable (first entry)
      4) Fallback to PAXG
    """
    env_candidates = [
        ("TRADING_SYMBOL", os.getenv("TRADING_SYMBOL")),
        ("SYMBOL", os.getenv("SYMBOL")),
    ]

    for source, value in env_candidates:
        symbol = _extract_symbol(value)
        if symbol:
            logger.info(f"Using trading symbol from environment variable {source}: {symbol}")
            return symbol

    config_path = project_root / "user_data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            pair_whitelist = config_data.get("exchange", {}).get("pair_whitelist") or []
            if pair_whitelist:
                symbol = _extract_symbol(pair_whitelist[0])
                if symbol:
                    logger.info(f"Using trading symbol from config pair_whitelist: {symbol}")
                    return symbol
        except Exception as exc:  # Broad catch to keep runner resilient
            logger.warning(f"Could not load trading symbol from config {config_path}: {exc}")
    else:
        logger.warning(f"Config file not found at {config_path}, skipping config-based symbol detection.")

    # Secondary fallback: a SYMBOLS env var (may be comma-delimited)
    symbols_env = os.getenv("SYMBOLS")
    symbol = _extract_symbol(symbols_env)
    if symbol:
        logger.info(f"Using trading symbol from environment variable SYMBOLS: {symbol}")
        return symbol

    fallback_symbol = "PAXG"
    logger.warning(f"Falling back to default trading symbol: {fallback_symbol}")
    return fallback_symbol


def run_avellaneda_param_calculation(hour_interval=0.25, ticker=None):
    """
    Executes the Avellaneda parameter calculation script with duplicate run protection.

    Args:
        hour_interval (float): Number of hours to wait between executions (default: 0.25 / 15 minutes)
        ticker (str, optional): Trading symbol override. If None, resolves from env/config.

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

    trading_symbol = ticker or _determine_trading_symbol(project_root, logger)
    logger.info(f"Target trading symbol for parameter calculation: {trading_symbol}")
    
    # Check if the calculation script exists
    if not script_path.exists():
        error_msg = f"Calculation script not found at: {script_path}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "symbol": trading_symbol
        }
    
    logger.info(f"Calculation script found: {script_path}")
    
    # Check for previous execution within the specified hour interval
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
            logger.info(f"Hour interval: {hour_interval}")

            time_since_last_run = current_time_utc - last_run_time_utc
            required_interval = timedelta(hours=hour_interval)

            logger.info(f"Time since last run: {time_since_last_run}")
            logger.info(f"Required interval: {required_interval}")

            # Skip if less than the specified hour interval has passed
            if time_since_last_run < required_interval:
                time_remaining = required_interval - time_since_last_run
                hours_remaining = int(time_remaining.total_seconds() // 3600)
                minutes_remaining = int((time_remaining.total_seconds() % 3600) // 60)

                skip_message = (f"Calculation already performed within the last {hour_interval} hour(s). "
                              f"Time remaining: {hours_remaining}h {minutes_remaining}m.")

                logger.info(f"EXECUTION SKIPPED: {skip_message}")

                return {
                    "status": "skipped",
                    "message": skip_message,
                    "last_run": last_run_time.isoformat(),
                    "hour_interval": hour_interval,
                    "time_remaining_minutes": round(time_remaining.total_seconds() / 60, 2),
                    "symbol": trading_symbol
                }
            else:
                logger.info(f"Required interval of {hour_interval} hour(s) has passed. Proceeding with calculation.")
                
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
            command = [sys.executable, str(script_path), trading_symbol]
            logger.info(f"Executing command: {' '.join(command)}")
            logger.info(f"Timeout set to: 3600 seconds")

            # Pass through environment variables for consistent paths
            env = os.environ.copy()
            # Ensure AVELLANEDA_PARAMS_DIR is set if not already
            if 'AVELLANEDA_PARAMS_DIR' not in env:
                # Default to scripts directory
                env['AVELLANEDA_PARAMS_DIR'] = str(project_root / "scripts")
                logger.info(f"Setting AVELLANEDA_PARAMS_DIR to: {env['AVELLANEDA_PARAMS_DIR']}")

            result = subprocess.run(command, capture_output=True, text=True, timeout=3600, env=env)
            
            logger.info(f"Subprocess completed with return code: {result.returncode}")
            
            # Log subprocess output regardless of success/failure
            log_subprocess_output(logger, result.stdout, result.stderr, "CALCULATION SCRIPT")
            
            if result.returncode == 0:
                # Update the lock file with current timestamp in UTC
                current_time = datetime.now(timezone.utc)
                lock_data = {
                    "last_run": current_time.isoformat(),
                    "script_path": str(script_path),
                    "execution_status": "success",
                    "symbol": trading_symbol
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
                    "symbol": trading_symbol,
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
                    "stderr": result.stderr,
                    "symbol": trading_symbol
                }
                
        finally:
            # Always restore the original working directory
            logger.info(f"Restoring working directory to: {original_cwd}")
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        timeout_message = "Calculation script timed out after 3600 seconds"
        logger.error(f"TIMEOUT: {timeout_message}")
        return {
            "status": "error",
            "message": timeout_message,
            "symbol": trading_symbol
        }
    except Exception as e:
        exception_message = f"Error executing calculation script: {str(e)}"
        logger.error(f"EXCEPTION: {exception_message}")
        return {
            "status": "error",
            "message": exception_message,
            "symbol": trading_symbol
        }


if __name__ == "__main__":
    """
    Run the function directly when script is executed
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run Avellaneda parameter calculation with configurable interval')
    parser.add_argument(
        '--hours',
        type=float,
        default=0.25,
        help='Number of hours to wait between executions (default: 0.25 / 15 minutes)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Trading symbol override (e.g., PAXG). Falls back to config/env if omitted.'
    )
    args = parser.parse_args()

    result = run_avellaneda_param_calculation(hour_interval=args.hours, ticker=args.symbol)
    
    # Set up logging for final output (reuse the same logger)
    logger, log_file_path = setup_logging()
    
    log_separator(logger, "EXECUTION SUMMARY")
    logger.info(f"Status: {result['status'].upper()}")
    logger.info(f"Message: {result['message']}")
    if result.get('symbol'):
        logger.info(f"Symbol: {result['symbol']}")
    
    if result.get('execution_time'):
        logger.info(f"Execution Time: {result['execution_time']}")
    
    if result.get('last_run'):
        logger.info(f"Last Run: {result['last_run']}")
    
    if result.get('current_date_utc'):
        logger.info(f"Current UTC Date: {result['current_date_utc']}")
        logger.info(f"Last Run UTC Date: {result.get('last_run_date_utc', 'N/A')}")
        logger.info(f"Minutes Since Midnight UTC: {result.get('minutes_since_midnight_utc', 'N/A')}")
    
    hour_interval = result.get('hour_interval', 0.25)
    logger.info(f"Execution Schedule:")
    logger.info(f"  • Every {hour_interval} hour(s) from last successful run")
    
    # Final subprocess output logging (if not already logged)
    if result.get('stdout') or result.get('stderr'):
        log_subprocess_output(logger, result.get('stdout'), result.get('stderr'), "FINAL OUTPUT")
    
    log_separator(logger, "EXECUTION COMPLETE")
    logger.info(f"Log file saved to: {log_file_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] in ['success', 'skipped'] else 1)

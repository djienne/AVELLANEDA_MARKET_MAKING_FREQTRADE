from pathlib import Path
import json
import sys
from datetime import datetime, timezone

def fraction_of_day_remaining_utc():
    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    
    # Get end of day (midnight) in UTC
    end_of_day = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    # Or more precisely, start of next day:
    # end_of_day = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate total seconds in a day
    total_seconds_in_day = 24 * 60 * 60
    
    # Calculate seconds remaining
    seconds_remaining = (end_of_day - now_utc).total_seconds()
    
    # Calculate fraction remaining
    fraction_remaining = seconds_remaining / total_seconds_in_day
    
    return fraction_remaining

def find_upwards(filename: str, start: Path, max_up: int = 10) -> Path:
    p = start.resolve()
    for _ in range(max_up + 1):
        candidate = p / filename
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError(f"Could not find {filename} from {start}")

def load_configs(start_dir: Path | None = None, max_up: int = 10):
    """
    Load Avellaneda parameters from JSON file, searching in multiple locations.
    
    Args:
        start_dir: Starting directory for search (defaults to current file's directory)
        max_up: Maximum levels to search upwards
        
    Returns:
        dict: Loaded parameters from JSON file
        
    Raises:
        FileNotFoundError: If the parameters file cannot be found
    """
    if start_dir is None:
        try:
            start_dir = Path(__file__).resolve().parent
        except NameError:  # e.g., interactive
            start_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()

    # List of potential file locations to check (in order of preference)
    search_locations = [
        "scripts/avellaneda_parameters_BTC.json",  # Most likely location
        "avellaneda_parameters_BTC.json",          # Root directory
        "user_data/strategies/avellaneda_parameters_BTC.json",  # Same directory
        "../scripts/avellaneda_parameters_BTC.json",  # One level up then scripts
        "../../scripts/avellaneda_parameters_BTC.json"  # Two levels up then scripts
    ]
    
    # First try to find the file using the existing upward search for each location
    for location in search_locations:
        try:
            params_file = find_upwards(location, start_dir, max_up)
            params_MM = json.loads(params_file.read_text(encoding="utf-8"))
            print(f"Successfully loaded parameters from: {params_file}")
            return params_MM
        except FileNotFoundError:
            continue
    
    # If upward search fails, try direct relative paths from start directory
    for location in search_locations:
        try:
            # Try relative to start directory
            params_path = start_dir / location
            if params_path.exists():
                params_MM = json.loads(params_path.read_text(encoding="utf-8"))
                print(f"Successfully loaded parameters from: {params_path}")
                return params_MM
                
            # Try relative to project root (go up to find project root)
            current = start_dir
            for _ in range(max_up):
                potential_root = current / location
                if potential_root.exists():
                    params_MM = json.loads(potential_root.read_text(encoding="utf-8"))
                    print(f"Successfully loaded parameters from: {potential_root}")
                    return params_MM
                if current.parent == current:
                    break
                current = current.parent
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {location}: {e}")
            continue
    
    # If all attempts fail, provide helpful error message
    raise FileNotFoundError(
        f"Could not find 'avellaneda_parameters_BTC.json' in any of these locations:\n"
        f"  - {chr(10).join([str(start_dir / loc) for loc in search_locations])}\n"
        f"Please ensure the file exists and is accessible."
    )


if __name__ == "__main__":

    params_MM = load_configs()

    print(params_MM)

    gamma = params_MM['optimal_parameters']['gamma']
    k = params_MM['market_data']['k']
    sigma = params_MM['market_data']['sigma']

    print(sigma,k,gamma)

    print(fraction_of_day_remaining_utc())
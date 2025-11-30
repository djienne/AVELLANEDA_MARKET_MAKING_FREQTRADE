
import sys
print("Hello from stdout")
print("Hello from stderr", file=sys.stderr)
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
    import pandas
    print(f"Pandas version: {pandas.__version__}")
    import arch
    print(f"Arch version: {arch.__version__}")
    import pyarrow
    print(f"Pyarrow version: {pyarrow.__version__}")
except Exception as e:
    print(f"Error importing: {e}")

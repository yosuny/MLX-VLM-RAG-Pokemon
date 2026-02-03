import mlx_vlm
from mlx_vlm import load
import inspect

print(f"mlx_vlm version: {mlx_vlm.__version__ if hasattr(mlx_vlm, '__version__') else 'unknown'}")
print(f"mlx_vlm file: {mlx_vlm.__file__}")

try:
    from mlx_vlm import fuse
    print("✅ mlx_vlm.fuse module exists via import!")
except ImportError:
    print("❌ mlx_vlm.fuse module NOT found via import.")

# Check utilities
import mlx_vlm.utils
print("mlx_vlm.utils attributes:", dir(mlx_vlm.utils))

# Check if model has fuse method (mock load not possible easily without weights, but let's check utils)

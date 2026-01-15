import mlx_vlm
import inspect

print("mlx_vlm version:", getattr(mlx_vlm, "__version__", "unknown"))
print("mlx_vlm dir:", dir(mlx_vlm))

try:
    from mlx_vlm import lora
    print("mlx_vlm.lora available")
    print("lora dir:", dir(lora))
except ImportError:
    print("mlx_vlm.lora NOT available")

try:
    import mlx_vlm.train
    print("mlx_vlm.train available")
except ImportError:
    print("mlx_vlm.train NOT available")

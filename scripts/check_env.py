import sys
import platform

print("Python:", sys.version)
print("Platform:", platform.platform())

packages = [
    "numpy",
    "pandas",
    "datasets",
    "modelscope",
    "transformers",
    "accelerate",
    "PIL",
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: {version}")
    except Exception as e:
        print(f"{pkg}: NOT FOUND ({e})")

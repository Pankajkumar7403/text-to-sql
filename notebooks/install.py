import subprocess
subprocess.run([
    "pip", "install", "--quiet",
    "unsloth",
    "peft>=0.12.0",
    "duckdb>=1.2.2",
    "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0",
], check=True)
subprocess.run([
    "pip", "install", "--quiet", "--upgrade", "--no-cache-dir", "--no-deps",
    "git+https://github.com/unslothai/unsloth.git",
], check=True)
print("Packages installed.")
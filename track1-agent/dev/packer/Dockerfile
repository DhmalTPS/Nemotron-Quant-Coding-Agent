FROM finance-bench-sandbox:latest
RUN pip install --no-cache-dir "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.5.1.zip#subdirectory=common" && qfbench2 submission pack --help >/dev/null && echo packer-ok

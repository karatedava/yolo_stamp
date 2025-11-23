rm -rf /dev/shm/* 2>/dev/null || true
python -c "import torch,gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true
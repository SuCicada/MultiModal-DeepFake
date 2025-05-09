import test2
import sys

# 设置参数
EXPID = "checkpoint"
HOST = "127.0.0.1"
PORT = "2"
NUM_GPU = 1

# 设置命令行参数
sys.argv = [
    "test2.py",
    "--config", "configs/test.yaml",
    "--output_dir", "results",
    "--launcher", "pytorch",
    "--rank", "0",
    "--log_num", EXPID,
    "--dist-url", f"tcp://{HOST}:1003{PORT}",
    "--token_momentum",
    "--world_size", str(NUM_GPU),
    "--test_epoch", "best"
]

# 调用main函数
test2.main()

#!/usr/bin/env python3
import torch
import time
import sys

# List of GPU device indices to occupy
gpu_indices = [6,7,8,9]  # modify as needed
# Fraction of total memory to allocate on each GPU
fraction = 0.95

# Keep references alive so memory stays allocated
buffers = []

for device in gpu_indices:
    try:
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory
        alloc_bytes = int(total_mem * fraction)
        # Allocate a 1-byte tensor of the desired size
        buf = torch.empty(alloc_bytes, dtype=torch.uint8, device=device)
        buffers.append(buf)
        print(f"✅ Occupied {fraction*100:.1f}% of GPU {device} ({total_mem // 1024**2} MiB total).")
    except RuntimeError as e:
        print(f"❌ Failed to occupy GPU {device}: {e}", file=sys.stderr)

print("\nAll done! Press Ctrl+C to free memory and exit.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\nReleasing GPU memory and exiting.")
    sys.exit(0)

# # devices = ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
# devices = ["cuda:1", "cuda:2", "cuda:8", "cuda:9"]
# # devices = ["cuda:7"]
# # devices = ["cuda:8"]
# import torch
# import threading
# import time
# import random

# def gpu_dynamic_workload(device: str):
#     """
#     Continuously alternates between different workload levels on the GPU
#     to produce a dynamic GPU utilization.
    
#     The function randomly selects one of the following modes on each
#     iteration:
#       - "idle": Sleep for a short period (simulate low usage)
#       - "light": Perform a small matrix multiplication repeatedly
#       - "moderate": Use a larger matrix multiplication workload
#       - "heavy": Perform heavy computations with a large matrix
#     """
#     # Set the current thread's GPU device.
#     torch.cuda.set_device(device)
#     print(f"[{device}] Starting dynamic workload.")

#     while True:
#         # Randomly choose a workload level.
#         workload_level = random.choice(["idle", "light", "moderate", "heavy"])

#         if workload_level == "idle":
#             sleep_time = random.uniform(1, 3)
#             print(f"[{device}] Idle mode: Sleeping for {sleep_time:.2f} seconds.")
#             time.sleep(sleep_time)
#         else:
#             # Configure parameters based on the workload level.
#             if workload_level == "light":
#                 matrix_size = 256*32
#                 duration = random.uniform(2, 4)
#             elif workload_level == "moderate":
#                 matrix_size = 512*32
#                 duration = random.uniform(3, 5)
#             elif workload_level == "heavy":
#                 matrix_size = 1024*32
#                 duration = random.uniform(4, 6)
            
#             # Allocate two matrices of the given size on the GPU.
#             print(f"[{device}] {workload_level.title()} load: Running computations with matrix size {matrix_size} for ~{duration:.2f} seconds.")
#             A = torch.rand(matrix_size, matrix_size, device=device)
#             B = torch.rand(matrix_size, matrix_size, device=device)

#             # Run the computation for a fixed duration.
#             start_time = time.time()
#             iteration = 0
#             while time.time() - start_time < duration:
#                 C = torch.mm(A, B)
#                 C = C + 1.0  # Additional operation to vary the workload.
#                 iteration += 1
#             print(f"[{device}] Completed {iteration} iterations during {workload_level} load.")

# def main():
#     if torch.cuda.device_count() < 2:
#         print("This script requires at least two GPUs. Exiting.")
#         return

#     threads = []

#     # Start a dedicated thread for each GPU.
#     for device in devices:
#         thread = threading.Thread(target=gpu_dynamic_workload, args=(device,))
#         thread.start()
#         threads.append(thread)

#     # Keep the main thread alive by joining the worker threads.
#     for thread in threads:
#         thread.join()

# if __name__ == "__main__":
#     main()
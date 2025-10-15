import numpy as np
import os

path = '/nas/datasets/LYX/PCMA/QPSK_16/1050/'


original_file_name = '1050000000_10000000__12000000_20250623170252_902610_0000.dat'
output_file_name = '/nas/datasets/yixin/PCMA/sim_data/splited_data.pth'

original_file = path + original_file_name
output_file = output_file_name

N = int(3072*100)

file_size = os.path.getsize(original_file)
print(f"Original file size: {file_size / (1024**2):.2f} MB")
total_samples_in_file = file_size // (2 * 2)
print(f"Total samples in original file: {total_samples_in_file}")

# 检查N是否超出文件范围
if N > total_samples_in_file:
    print(f"Warning: N ({N}) is larger than the total samples in the file ({total_samples_in_file}).")
    print(f"Will read all {total_samples_in_file} samples instead.")
    N = total_samples_in_file

print(f"\nReading first {N} samples...")
mmap_data = np.memmap(original_file, dtype=np.int16, mode='r', shape=(N * 2,))

data_reshaped = mmap_data.reshape(-1, 2)

complex_data_to_save = data_reshaped[:, 0].astype(np.float32) + 1j * data_reshaped[:, 1].astype(np.float32)

print(f"Successfully read {len(complex_data_to_save)} complex samples.")
print(f"Data type for saving: {complex_data_to_save.dtype}")

real_part = complex_data_to_save.real
imag_part = complex_data_to_save.imag

# 将它们堆叠成一个 (N, 2) 的数组
data_to_write = np.stack((real_part, imag_part), axis=1)

final_data_to_write = data_to_write.astype(np.int16)
print(f"\nSaving {len(final_data_to_write)} samples to '{output_file}'...")

final_data_to_write.tofile(output_file)

# 验证
new_file_size = os.path.getsize(output_file)
expected_new_size = N * 2 * 2

print(f"\n--- Verification ---")
print(f"New file saved at: {os.path.abspath(output_file)}")
print(f"New file size: {new_file_size} bytes")
print(f"Expected new file size: {expected_new_size} bytes")

if new_file_size == expected_new_size:
    print("✅ File size matches expectation.")

    print("Verifying content of the new file...")

    verify_mmap = np.memmap(output_file, dtype=np.int16, mode='r', shape=(5 * 2,))
    verify_data = verify_mmap.reshape(-1, 2)
    verify_complex = verify_data[:, 0] + 1j * verify_data[:, 1]
    
    # 与原始数据的前5个样本比较
    original_complex_samples = complex_data_to_save[:5]
    
    print("First 5 samples from the new file:", verify_complex)
    print("First 5 samples from the original read:", original_complex_samples)
    
    if np.allclose(verify_complex, original_complex_samples):
        print("✅ Content of the new file is correct!")
    else:
        print("❌ Content mismatch!")
else:
    print("❌ File size does NOT match expectation!")


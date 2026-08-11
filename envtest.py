import torch, time

'''
建議把 torch.set_num_threads() 設成你的實體核心數
（雙核心 i7，含超執行緒是 4 threads，直接用預設值通常就好，不用手動調）。
'''

print(torch.__version__)
print(torch.get_num_threads())   # CPU 推論會吃多核心，先確認有抓到全部核心

'''
CPU 版推論主要靠多執行緒平行運算
torch.get_num_threads() 太低的話矩陣運算會偏慢。

手動測一下调高 torch.set_num_threads(n) 會不會更快：
'''

x = torch.randn(512, 512)
y = torch.randn(512, 512)

for n in [2, 4]:
    torch.set_num_threads(n)
    t0 = time.perf_counter()
    for _ in range(200):
        z = x @ y
    print(f"threads={n}: {time.perf_counter() - t0:.4f}s")
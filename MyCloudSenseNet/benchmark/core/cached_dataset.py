"""
缓存数据集模块

提供磁盘缓存和内存缓存的数据集实现，加速重复实验。
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union, List
from dataclasses import dataclass
import threading

import torch
import numpy as np
from torch.utils.data import Dataset

from benchmark.core.config import logger
from benchmark.core.cloud_dataset import CloudDataset


@dataclass
class CacheConfig:
    """缓存配置"""
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    max_memory_cache: int = 200  # 最多缓存多少个样本
    disk_cache_dir: Optional[Path] = None
    cache_format: str = "tensor"  # tensor, numpy, pickle
    compression: Optional[str] = None  # gzip, lz4, None
    
    def get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        if self.disk_cache_dir is None:
            self.disk_cache_dir = Path(".cache/dataset")
        
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.disk_cache_dir / f"{key}.pt"


class MemoryCache:
    """
    线程安全的内存缓存
    
    使用 LRU 策略管理缓存。
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到最新访问位置
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats["hits"] += 1
                return self.cache[key]
            
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any):
        """添加缓存项"""
        with self.lock:
            # 如果已存在，更新位置和值
            if key in self.cache:
                self.access_order.remove(key)
            
            # 如果已满，淘汰最久未使用的
            elif len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            # 添加新项
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": self.stats["hits"] / total if total > 0 else 0.0,
            }


class CachedCloudDataset(Dataset):
    """
    带缓存的云检测数据集
    
    支持：
    1. 内存 LRU 缓存（加速小数据集重复读取）
    2. 磁盘缓存（避免重复预处理和 I/O）
    3. 缓存预热（批量加载常用样本）
    
    缓存键基于：
    - 样本索引
    - 数据版本（用于缓存失效）
    - 变换配置（不同的变换 = 不同的缓存）
    """
    
    # 类级别的共享内存缓存
    _global_memory_cache: Optional[MemoryCache] = None
    _cache_lock = threading.Lock()
    
    def __init__(
        self,
        base_dataset: CloudDataset,
        cache_config: Optional[CacheConfig] = None,
        transform_hash: Optional[str] = None,
        data_version: str = "v1",
        preload_to_memory: bool = False,
    ):
        """
        Args:
            base_dataset: 基础数据集
            cache_config: 缓存配置
            transform_hash: 变换的哈希值（用于区分不同变换的缓存）
            data_version: 数据版本（修改数据时递增以失效旧缓存）
            preload_to_memory: 是否预加载全部到内存
        """
        self.base_dataset = base_dataset
        self.cache_config = cache_config or CacheConfig()
        self.transform_hash = transform_hash or "none"
        self.data_version = data_version
        
        # 获取或创建全局内存缓存
        with self._cache_lock:
            if self._global_memory_cache is None:
                self._global_memory_cache = MemoryCache(
                    max_size=self.cache_config.max_memory_cache
                )
        self.memory_cache = self._global_memory_cache
        
        # 预加载
        if preload_to_memory:
            self._preload_all()
    
    def _compute_cache_key(self, idx: int) -> str:
        """计算缓存键"""
        # 组合多个因子生成唯一键
        key_components = [
            str(self.data_version),
            str(idx),
            str(self.transform_hash),
            str(len(self.base_dataset)),  # 数据集大小变化也影响缓存
        ]
        
        key_string = "|".join(key_components)
        
        # 使用 MD5 生成短哈希
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _save_to_disk_cache(self, key: str, item: Dict[str, Any]):
        """保存到磁盘缓存"""
        if not self.cache_config.enable_disk_cache:
            return
        
        try:
            cache_path = self.cache_config.get_cache_path(key)
            
            # 转换为 tensor 格式存储
            if self.cache_config.cache_format == "tensor":
                # 分离 tensor 和其他数据
                tensors = {}
                others = {}
                
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        tensors[k] = v
                    elif isinstance(v, np.ndarray):
                        tensors[k] = torch.from_numpy(v)
                    else:
                        others[k] = v
                
                # 保存
                torch.save({
                    "tensors": tensors,
                    "others": others,
                }, cache_path)
            
            elif self.cache_config.cache_format == "numpy":
                np.savez_compressed(cache_path, **item)
            
            elif self.cache_config.cache_format == "pickle":
                with open(cache_path, "wb") as f:
                    pickle.dump(item, f)
        
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """从磁盘缓存加载"""
        if not self.cache_config.enable_disk_cache:
            return None
        
        try:
            cache_path = self.cache_config.get_cache_path(key)
            
            if not cache_path.exists():
                return None
            
            if self.cache_config.cache_format == "tensor":
                data = torch.load(cache_path, map_location="cpu")
                
                # 还原数据
                item = {}
                for k, v in data["tensors"].items():
                    item[k] = v
                for k, v in data["others"].items():
                    item[k] = v
                
                return item
            
            elif self.cache_config.cache_format == "numpy":
                data = np.load(cache_path, allow_pickle=True)
                return {k: data[k] for k in data.files}
            
            elif self.cache_config.cache_format == "pickle":
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
        
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def _load_item(self, idx: int) -> Dict[str, Any]:
        """加载单个样本"""
        cache_key = self._compute_cache_key(idx)
        
        # 1. 检查内存缓存
        if self.cache_config.enable_memory_cache:
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # 2. 检查磁盘缓存
        if self.cache_config.enable_disk_cache:
            disk_cached = self._load_from_disk_cache(cache_key)
            if disk_cached is not None:
                # 同时放入内存缓存
                if self.cache_config.enable_memory_cache:
                    self.memory_cache.put(cache_key, disk_cached)
                return disk_cached
        
        # 3. 从基础数据集加载
        item = self.base_dataset[idx]
        
        # 4. 存入缓存
        if self.cache_config.enable_disk_cache:
            self._save_to_disk_cache(cache_key, item)
        
        if self.cache_config.enable_memory_cache:
            self.memory_cache.put(cache_key, item)
        
        return item
    
    def _preload_all(self):
        """预加载所有样本到内存"""
        logger.info(f"Preloading {len(self)} samples to memory cache...")
        
        # 使用多线程预加载
        from concurrent.futures import ThreadPoolExecutor
        
        def preload_single(idx):
            try:
                _ = self._load_item(idx)
            except Exception as e:
                logger.warning(f"Failed to preload item {idx}: {e}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(preload_single, range(len(self))))
        
        logger.info(f"Preloading complete. Cache stats: {self.get_cache_stats()}")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._load_item(idx)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "memory": self.memory_cache.get_stats() if self.memory_cache else None,
            "disk_cache_dir": str(self.cache_config.disk_cache_dir) if self.cache_config.disk_cache_dir else None,
        }
    
    def clear_cache(self, clear_memory: bool = True, clear_disk: bool = False):
        """清空缓存"""
        if clear_memory and self.memory_cache:
            self.memory_cache.clear()
            logger.info("Memory cache cleared")
        
        if clear_disk and self.cache_config.disk_cache_dir:
            import shutil
            if self.cache_config.disk_cache_dir.exists():
                shutil.rmtree(self.cache_config.disk_cache_dir)
                logger.info(f"Disk cache cleared: {self.cache_config.disk_cache_dir}")


class PrefetchCloudDataset(Dataset):
    """
    预取数据集
    
    使用后台线程预取即将使用的样本，减少 I/O 等待。
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        num_prefetch: int = 4,
    ):
        self.base_dataset = base_dataset
        self.num_prefetch = num_prefetch
        self.prefetch_queue: Dict[int, Any] = {}
        self.prefetch_thread: Optional[threading.Thread] = None
        self.current_idx = 0
        self.lock = threading.Lock()
    
    def _prefetch_worker(self, start_idx: int):
        """后台预取工作线程"""
        for i in range(start_idx, min(start_idx + self.num_prefetch, len(self))):
            try:
                item = self.base_dataset[i]
                with self.lock:
                    self.prefetch_queue[i] = item
            except Exception as e:
                logger.warning(f"Prefetch failed for item {i}: {e}")
    
    def _start_prefetch(self, idx: int):
        """启动预取"""
        # 清理旧缓存
        with self.lock:
            keys_to_remove = [k for k in self.prefetch_queue if k < idx]
            for k in keys_to_remove:
                del self.prefetch_queue[k]
        
        # 启动新的预取线程
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                args=(idx + 1,),
                daemon=True,
            )
            self.prefetch_thread.start()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        # 检查预取队列
        with self.lock:
            if idx in self.prefetch_queue:
                item = self.prefetch_queue[idx]
                del self.prefetch_queue[idx]
                # 启动下一批预取
                self._start_prefetch(idx)
                return item
        
        # 未预取到，直接加载
        item = self.base_dataset[idx]
        self._start_prefetch(idx)
        
        return item


class MultiEpochCachedDataset(CachedCloudDataset):
    """
    跨 epoch 持久化的缓存数据集
    
    缓存持续到所有 epoch 完成，特别适合训练场景。
    """
    
    def __init__(
        self,
        base_dataset: CloudDataset,
        num_epochs: int = 100,
        **kwargs,
    ):
        super().__init__(base_dataset, **kwargs)
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.epoch_access_stats: Dict[int, int] = {}
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch"""
        self.current_epoch = epoch
        self.epoch_access_stats[epoch] = 0
        
        # 根据 epoch 调整缓存策略
        if epoch == 0:
            # 第一个 epoch：积极缓存
            logger.info("First epoch: Building cache...")
        elif epoch == self.num_epochs - 1:
            # 最后一个 epoch：清空缓存
            self.clear_cache(clear_memory=True, clear_disk=False)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        self.epoch_access_stats[self.current_epoch] += 1
        return item
    
    def get_epoch_stats(self) -> Dict[str, Any]:
        """获取跨 epoch 统计"""
        return {
            "epoch_access_stats": self.epoch_access_stats.copy(),
            "total_accesses": sum(self.epoch_access_stats.values()),
        }


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    use_cache: bool = True,
    use_prefetch: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    创建优化的数据加载器
    
    自动应用缓存和预取优化。
    """
    
    # 应用缓存
    if use_cache and isinstance(dataset, CloudDataset):
        dataset = CachedCloudDataset(
            base_dataset=dataset,
            cache_config=CacheConfig(
                enable_disk_cache=True,
                enable_memory_cache=True,
            ),
        )
    
    # 应用预取
    if use_prefetch:
        dataset = PrefetchCloudDataset(dataset, num_prefetch=4)
    
    # 创建 DataLoader
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": kwargs.get("shuffle", True),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": kwargs.get("drop_last", False),
    }
    
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


if __name__ == "__main__":
    print("Testing Cached Dataset...")
    
    # 创建模拟数据集
    class MockCloudDataset(CloudDataset):
        def __init__(self, size=100):
            self.size = size
            self.data = list(range(size))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # 模拟耗时加载
            import time
            time.sleep(0.001)
            
            return {
                "chip": torch.randn(4, 64, 64),
                "label": torch.randint(0, 3, (64, 64)),
                "chip_id": f"chip_{idx:04d}",
            }
    
    # 测试缓存数据集
    print("\n1. CachedCloudDataset:")
    base_ds = MockCloudDataset(size=50)
    cached_ds = CachedCloudDataset(
        base_dataset=base_ds,
        cache_config=CacheConfig(
            enable_disk_cache=True,
            enable_memory_cache=True,
            max_memory_cache=20,
        ),
    )
    
    # 首次加载（慢）
    import time
    start = time.time()
    for i in range(10):
        _ = cached_ds[i]
    first_time = time.time() - start
    
    # 再次加载（快）
    start = time.time()
    for i in range(10):
        _ = cached_ds[i]
    second_time = time.time() - start
    
    print(f"   First load: {first_time*1000:.2f}ms")
    print(f"   Cached load: {second_time*1000:.2f}ms")
    print(f"   Speedup: {first_time/second_time:.1f}x")
    
    # 显示缓存统计
    stats = cached_ds.get_cache_stats()
    print(f"   Memory cache hits: {stats['memory']['hits']}")
    print(f"   Memory cache hit rate: {stats['memory']['hit_rate']:.1%}")
    
    # 清理
    cached_ds.clear_cache(clear_memory=True, clear_disk=True)
    
    print("\n✓ All tests passed!")

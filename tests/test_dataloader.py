import sys
import os
import torch.distributed as dist
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.worker.data import build_dataloader


class IndexDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)

    args = parser.parse_args()

    rank = args.rank
    world_size = args.world_size

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
    )

    dataset = IndexDataset(20)

    dataloader, sampler = build_dataloader(
        dataset, batch_size=4, rank=rank, world_size=world_size
    )

    sampler.set_epoch(0)

    print(f"\nWorker {rank} sees:")

    for batch in dataloader:
        print(batch.tolist())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_dataloader(
    dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
):
    """
    Build distributed dataloader so each worker gets unique data.

    Returns:
        dataloader, sampler
    """

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return dataloader, sampler

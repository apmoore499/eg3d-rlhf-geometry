import torch


class CollateVariableShapeBatch:
    """Collate fn for training data whose per-sample tensors may differ in
    shape and count across batches.

    Each dataset item is a ``(key, payload)`` tuple where ``payload`` is a
    dict with:
      - ``files``: list of tensors stacked along a new leading axis,
      - ``batch_len``: int, length of that sample's file list,
      - ``ordered_seeds``: 1-D tensor of seed ids.

    A fresh instance is constructed per batch (no persistent buffer reuse),
    so successive batches are free to have different shapes. ``pin_memory``
    is implemented so this works as a custom batch type with a pin-memory
    dataloader.
    """

    def __init__(self, batch):
        batches = [b[1] for b in batch]
        self.file_batch = torch.cat(
            [torch.cat([bb.unsqueeze(0) for bb in b["files"]], dim=0).unsqueeze(0) for b in batches],
            0,
        )
        self.lens_batch = torch.cat([torch.tensor(b["batch_len"], device=self.file_batch.device).unsqueeze(0) for b in batches])
        self.ordered_seeds = torch.cat([b["ordered_seeds"].unsqueeze(0) for b in batches])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.file_batch = self.file_batch.pin_memory()
        self.lens_batch = self.lens_batch.pin_memory()
        self.ordered_seeds = self.ordered_seeds.pin_memory()
        return self

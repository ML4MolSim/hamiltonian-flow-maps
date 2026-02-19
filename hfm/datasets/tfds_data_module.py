from hfm.datasets.tfds_dataset import TfdsDataset


class DummyDataset:
    def __init__(self, split, tfds):
        self.split = split
        self.tfds = tfds
    
    def next_epoch(self, rng, batch_size):
        return self.tfds.next_epoch(batch_size, split=self.split)

    def get_example_batch(self, rng, batch_size):
        return self.tfds.get_example_batch(batch_size, split=self.split)

    def __len__(self):
        return self.tfds.get_len(self.split)


class TfdsDataModule:
    """Provides some functions for splitting tfds datasets."""
    def __init__(self, split_train, split_val, split_test, name="", **kwargs):
        self.name = name
        
        self.tfds = TfdsDataset(**kwargs)

        self.train_dataset = DummyDataset(split=split_train, tfds=self.tfds)
        self.val_dataset = DummyDataset(split=split_val, tfds=self.tfds)
        self.test_dataset = DummyDataset(split=split_test, tfds=self.tfds)

    def shutdown(self):
        self.tfds.shutdown()

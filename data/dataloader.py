import torch
import torchvision.transforms as transforms
import numpy as np
import threading
from queue import Queue
import time
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class ClassBatchSampler(object):
    def __init__(self, cls_idx, batch_size, drop_last=True):
        self.samplers = []
        for indices in cls_idx:
            n_ex = len(indices)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size=min(n_ex, batch_size), drop_last=drop_last
            )
            self.samplers.append(iter(_RepeatSampler(batch_sampler)))

    def __iter__(self):
        while True:
            for sampler in self.samplers:
                yield next(sampler)

    def __len__(self):
        return len(self.samplers)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device("cpu"):
            self.device = "cpu"
        else:
            self.device = "cuda"

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target


class ClassDataLoader(MultiEpochsDataLoader):
    """Basic class loader (might be slow for processing data)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(self.dataset)):
            self.cls_idx[self.dataset.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(
            self.cls_idx, self.batch_size, drop_last=True
        )

        self.cls_targets = torch.tensor(
            [np.ones(self.batch_size) * c for c in range(self.nclass)],
            dtype=torch.long,
            requires_grad=False,
            device="cuda",
        )

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.dataset[i][0] for i in indices])
        target = torch.tensor([self.dataset.targets[i] for i in indices])
        return data.cuda(), target.cuda()

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class ClassMemDataLoader:
    """Class loader with data on GPUs"""

    def __init__(self, dataset, batch_size, drop_last=False, device="cuda"):
        self.device = device
        self.batch_size = batch_size

        self.dataset = dataset
        self.data = [d[0].to(device) for d in dataset]  # uint8 data
        self.targets = torch.tensor(dataset.targets, dtype=torch.long, device=device)

        sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(dataset))])
        self.batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )
        self.iterator = iter(_RepeatSampler(self.batch_sampler))

        self.nclass = dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(
            self.cls_idx, self.batch_size, drop_last=True
        )
        self.cls_targets = torch.tensor(
            [np.ones(batch_size) * c for c in range(self.nclass)],
            dtype=torch.long,
            requires_grad=False,
            device=self.device,
        )

        self.convert = None
        if self.data[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print(self.targets[indices])
        return data, self.cls_targets[c]

    def sample(self):
        indices = next(self.iterator)
        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)
        target = self.targets[indices]

        return data, target

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            data, target = self.sample()
            yield data, target


class ClassPartMemDataLoader(MultiEpochsDataLoader):
    """Class loader for ImageNet-100 with multi-processing.
    This loader loads target subclass samples on GPUs
    while can loading full training data from storage.
    """

    def __init__(self, subclass_list, real_to_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.mem_cls = subclass_list
        self.real_to_idx = real_to_idx

        self.cls_idx = [[] for _ in range(self.nclass)]
        idx = 0
        self.data_mem = []
        print("Load target class data on memory..")
        for i in range(len(self.dataset)):
            c = self.dataset.targets[i]
            if c in self.mem_cls:
                self.data_mem.append(self.dataset[i][0].cuda())
                self.cls_idx[c].append(idx)
                idx += 1

        if self.data_mem[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)
        print(f"Subclass: {subclass_list}, {len(self.data_mem)}")

        class_batch_size = 64
        self.class_sampler = ClassBatchSampler(
            [self.cls_idx[c] for c in subclass_list], class_batch_size, drop_last=True
        )
        self.cls_targets = torch.tensor(
            [np.ones(class_batch_size) * c for c in range(self.nclass)],
            dtype=torch.long,
            requires_grad=False,
            device="cuda",
        )

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            idx = self.real_to_idx[c]
            indices = next(self.class_sampler.samplers[idx])

        data = torch.stack([self.data_mem[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print([self.dataset.targets[i] for i in self.slct[indices]])
        return data, self.cls_targets[c]

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class AsyncLoader:
    def __init__(self, loader_real, class_list, batch_size, device, num_Q=10):
        self.loader_real = loader_real  # The actual data loader
        self.batch_size = batch_size  # Batch size
        self.device = device  # Device (e.g., CPU or GPU)
        self.class_list = class_list  # List of classes
        self.nclass = len(class_list)  # Number of classes
        self.queue = Queue(maxsize=num_Q)  # Buffer queue
        self.current_index = 0  # Current class index
        self.stop_event = threading.Event()  # Stop flag for the background thread
        self.thread = threading.Thread(
            target=self._load_data, daemon=True
        )  # Background thread to load data
        self.thread.start()

    def _load_data(self):
        while not self.stop_event.is_set():
            if not self.queue.full():  # If the queue is not full
                # Current class
                current_class = self.class_list[self.current_index]
                # Load data
                img, img_label = self.loader_real.class_sample(
                    current_class, self.batch_size
                )
                img, img_label = img.to(self.device), img_label.to(
                    self.device
                )  # Move data to the device
                # Put data into the queue
                self.queue.put((img_label, img))
                # Update class index
                self.current_index = (self.current_index + 1) % self.nclass
            else:
                time.sleep(0.01)  # Wait briefly if the buffer is full

    def class_sample(self, c):
        """Get data of the specified class"""
        while True:
            img_label, img = self.queue.get()
            if img_label[0] == c:  # If the label matches the desired class
                return img, img_label
            else:
                # If not the target class, put the data back into the queue
                self.queue.put((img_label, img))

    def stop(self):
        """Stop the asynchronous data loading thread"""
        self.stop_event.set()
        self.thread.join()


class ImageNetMemoryDataLoader:
    def __init__(self, load_dir=None, debug=False, class_list=None):
        self.class_list = class_list  # List of classes to load
        self.load_dir = load_dir  # Directory to load data from
        self.debug = debug  # Whether to enable debug mode
        self.categorized_data = []  # List to store categorized data
        self.target_to_class_data = {}  # New: Maps target to class data
        self._load_categorized_data()  # Load the categorized data

    def _load_categorized_data(self):
        if self.load_dir is None:
            return None
        categorized_data = []
        file_list = sorted([f for f in os.listdir(self.load_dir) if f.endswith(".pt")])

        # Filter files based on class_list
        if self.class_list is not None:
            file_list = [
                f
                for f in file_list
                if int(f.split("_")[1].split(".")[0]) in self.class_list
            ]

        if self.debug:
            file_list = file_list[:1]  # In debug mode, only load the first file
            print(f"Debug mode enabled: only loading {file_list}")

        def load_file(file_name):
            file_path = os.path.join(self.load_dir, file_name)
            result = torch.load(file_path)

            # Check for uniqueness
            unique_targets = torch.unique(result["targets"])
            if len(unique_targets) != 1:
                raise ValueError(
                    f"File {file_name} contains multiple labels: {unique_targets.tolist()}"
                )

            # Check consistency between filename and label
            file_label = int(file_name.split("_")[1].split(".")[0])
            if unique_targets.item() != file_label:
                raise ValueError(
                    f"File {file_name} label {file_label} does not match targets {unique_targets.item()}"
                )
            return result

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(
                tqdm(
                    executor.map(load_file, file_list),
                    desc="Loading Categorized Data",
                    total=len(file_list),
                )
            )

        # Create a mapping from target to class data
        for result in results:
            categorized_data.append(result)
            target = torch.unique(result["targets"]).item()  # Get the unique target
            self.target_to_class_data[target] = (
                result  # Map target to corresponding class data
            )

        self.categorized_data = categorized_data

    def class_sample(self, c, batch_size=256):
        if c not in self.target_to_class_data:
            raise ValueError(f"Target {c} is not in the loaded dataset")

        # Retrieve the corresponding class data
        class_data = self.target_to_class_data[c]
        data, targets = class_data["data"], class_data["targets"]

        # Check that c matches the first target in the class data
        if c != targets[0].item():  # Convert to integer for comparison
            raise ValueError(
                f"Mismatch: Input target {c} does not match the first target in class_data {targets[0].item()}"
            )

        # Randomly sample
        indices = torch.randperm(len(data))[:batch_size]
        data = data[indices].to("cuda")  # Move to GPU
        targets = targets[indices].to("cuda")  # Move to GPU
        return data, targets  # Ensure targets are also on GPU

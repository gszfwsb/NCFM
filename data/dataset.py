import torch
import torchvision.datasets as datasets
from data.dataset_statistics import IMG_EXTENSIONS, STDS, MEANS


class Data:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        self.n_pool = len(X_train)

    def get_class_data(self, c):
        idxs = torch.arange(self.n_pool)
        idxs_c = torch.where(self.Y_train[idxs] == c)
        idxs = idxs[idxs_c[0]]
        dst_train = Dataset(self.X_train[idxs], self.Y_train[idxs])
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=256, shuffle=False, num_workers=0
        )
        return idxs, trainloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        # images: NxCxHxW tensor
        self.images = images.float()
        self.targets = labels

    def __getitem__(self, index):
        sample = self.images[index]
        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().float()
        self.targets = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class ImageFolder_mtt(datasets.DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        load_memory=False,
        load_transform=None,
        type="none",
        slct_type="random",
        ipc=-1,
    ):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder_mtt, self).__init__(
            root,
            loader,
            self.extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        # Override
        self.nclass = 10
        self.classes, self.class_to_idx = self.find_subclasses(type=type)

        self.samples = datasets.folder.make_dataset(
            self.root, self.class_to_idx, self.extensions, is_valid_file
        )

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, type="none"):
        """Finds the class folders in a dataset."""
        classes = []
        # ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']
        if type != "none":
            with open("./imagenet_subset/class{}.txt".format(type), "r") as f:
                class_name = f.readlines()
        for c in class_name:
            c = c.split("\n")[0]
            classes.append(c)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == self.nclass

        return classes, class_to_idx

    def _subset(self, slct_type="random", ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc < min_class

        if slct_type == "random":
            indices = np.arange(n)
        else:
            raise AssertionError(f"selection type does not exist!")

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory"""
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end="\r")

        print(" " * 50, end="\r")
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        load_memory=False,
        load_transform=None,
        nclass=100,
        phase=0,
        slct_type="random",
        ipc=-1,
        seed=-1,
    ):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder, self).__init__(
            root,
            loader,
            self.extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(
                nclass=nclass, phase=phase, seed=seed
            )
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(
            self.root, self.class_to_idx, self.extensions, is_valid_file
        )
        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)
        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == 0:
            with open("./imagenet_subset/class100.txt", "r") as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split("\n")[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass
        return classes, class_to_idx

    def _subset(self, slct_type="random", ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)
        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc < min_class
        if slct_type == "random":
            indices = np.arange(n)
        else:
            raise AssertionError(f"selection type does not exist!")
        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break
        return samples_subset

    def _load_images(self, transform=None):
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end="\r")
        print(" " * 50, end="\r")
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

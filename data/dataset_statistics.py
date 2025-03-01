
# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]
MEANS['tinyimagenet'] = [0.485, 0.456, 0.406]
STDS['tinyimagenet'] = [0.229, 0.224, 0.225]


# ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']
MEANS['imagenette'] = [0.485, 0.456, 0.406]
STDS['imagenette'] = [0.229, 0.224, 0.225]
MEANS['imagewoof'] = [0.485, 0.456, 0.406]
STDS['imagewoof'] = [0.229, 0.224, 0.225]
MEANS['imagemeow'] = [0.485, 0.456, 0.406]
STDS['imagemeow'] = [0.229, 0.224, 0.225]
MEANS['imagesquawk'] = [0.485, 0.456, 0.406]
STDS['imagesquawk'] = [0.229, 0.224, 0.225]
MEANS['imagefruit'] = [0.485, 0.456, 0.406]
STDS['imagefruit'] = [0.229, 0.224, 0.225]
MEANS['imageyellow'] = [0.485, 0.456, 0.406]
STDS['imageyellow'] = [0.229, 0.224, 0.225]

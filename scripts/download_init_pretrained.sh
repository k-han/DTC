path="data/experiments/pretrained/"
mkdir -p $path
cd $path

for file in "resnet18_cifar10_classif_5.pth" "vgg6_cifar100_classif_80.pth" "resnet18_svhn_classif_5.pth" "vgg6_omniglot_proto.pth" "resnet18_imagenet_classif_800.pth" "resnet18_imagenet_classif_882_ICLR18.pth" "resnet18_imagenet32_classif_1000.pth"; do
    wget http://www.robots.ox.ac.uk/~vgg/research/DTC/data/experiments/pretrained/${file}
done

# Back to root directory
cd ../

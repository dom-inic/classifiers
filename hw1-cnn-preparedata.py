import os
import torchvision
import tarfile

def download_data(url):
    torchvision.datasets.utils.download_url(url, '.')
    print("Unzipping data...")
    with tarfile.open('./cifar10.tgz', 'r:gz') as f:
        f.extractall(path='./data')
    print("Unzipping done!")

def show_classes():
    data_dir = './data/cifar10'
    print("Data Directories: ", os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print("Classes: ", classes)

if __name__ == "__main__":
    download_data("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz")
    show_classes()

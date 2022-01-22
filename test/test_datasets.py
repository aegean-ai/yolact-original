import torchvision.datasets as dset
import torchvision.transforms as transforms
import typing

def coco_dataset(images_directory: str, annotation_file: str):

    cap = dset.CocoCaptions(root = images_directory,
                            annFile = annotation_file,
                            transform=transforms.ToTensor())

    print('Number of samples: ', len(cap))
    return len(cap)

    
def test_coco_dataset():
    assert coco_dataset(images_directory='/data/coco/images', annotation_file='/data/coco/captions.json') == 82783


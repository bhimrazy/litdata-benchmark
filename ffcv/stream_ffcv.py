from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout, Convert, ModuleWrapper
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
import torch
import torch.nn as nn

# Random resized crop
decoder = RandomResizedCropRGBImageDecoder((224, 224))
class ScaleBy255(nn.Module):
    def forward(self, x):
        return x / 255.

# Data decoding and augmentation
image_pipeline = [decoder, ToTensor(), ToTorchImage(), Convert(torch.float32), ModuleWrapper(ScaleBy255())]
label_pipeline = [IntDecoder(), ToTensor()]

# Pipeline for each data field
pipelines = {"image": image_pipeline, "label": label_pipeline}

# Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
loader = Loader(
    "data/imagenet-1m-ffcv/train_256_1.0_90.ffcv",
    batch_size=16,
    num_workers=32,
    order=OrderOption.SEQUENTIAL,
    pipelines=pipelines,
)

batch = next(iter(loader))
print("Batch:", batch[0].shape, batch[1].shape)
print("Batch:", batch[0].dtype, batch[1].dtype)


# rest of training / validation proceeds identically
print("Starting to load data...", loader)

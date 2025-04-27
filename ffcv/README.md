⚡ main ~/ffcv-imagenet export IMAGENET_DIR=/teamspace/studios/this_studio/data/imagenet-1m-raw
⚡ main ~/ffcv-imagenet export WRITE_DIR=/teamspace/studios/this_studio/data/imagenet-1m-ffcv
⚡ main ~/ffcv-imagenet sh write_imagenet.sh 256 1.0 90
Writing ImageNet train dataset to /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_1.0_90.ffcv
┌ Arguments defined────────┬────────────────────────────────────────────────────────────────────────────┐
│ Parameter                │ Value                                                                      │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ cfg.dataset              │ imagenet                                                                   │
│ cfg.split                │ train                                                                      │
│ cfg.data_dir             │ /teamspace/studios/this_studio/data/imagenet-1m-raw/train                  │
│ cfg.write_path           │ /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_1.0_90.ffcv │
│ cfg.write_mode           │ proportion                                                                 │
│ cfg.max_resolution       │ 256                                                                        │
│ cfg.num_workers          │ 16                                                                         │
│ cfg.chunk_size           │ 100                                                                        │
│ cfg.jpeg_quality         │ 90.0                                                                       │
│ cfg.subset               │ -1                                                                         │
│ cfg.compress_probability │ 1.0                                                                        │
└──────────────────────────┴────────────────────────────────────────────────────────────────────────────┘
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1281167/1281167 [08:18<00:00, 2571.13it/s]




# Experiments Setup

Create a data dir eg: `data` and create three subfolders for diff purpose
- imagenet-1m-ffcv
- imagenet-1m-litdata
- imagenet-1m-raw

Copy the imagenet raw dataset to the data/imagenet-1m-raw 
```sh
s5cmd cp "s3://imagenet-1m-template/raw/train/*" data/imagenet-1m-raw/train
```

Convert the imagaenet raw dataset to be used as the imagefolder by converting the original subfolders present in imaganet raw train as class index subfolders
```sh 
 python convert_imagenet_to_pytorch_style.py --data_dir data/imagenet-1m-raw/train 
 ```

 Prepare the ffcv dataset:
 Install ffcv and it's dependencies
 ```sh
 sh install-ffcv
 ```

 Clone `https://github.com/libffcv/ffcv-imagenet` and prep dataset preparation
 ```sh
git clone https://github.com/libffcv/ffcv-imagenet
pip install -r requirements.txt
 ```

Run prep script for ffcv dataset
```sh
# Required environmental variables for the script:
export IMAGENET_DIR=/teamspace/studios/this_studio/data/imagenet-1m-raw
export WRITE_DIR=/teamspace/studios/this_studio/data/imagenet-1m-ffcv

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
# ./write_imagenet.sh 500 0.50 90
cd ffcv-imagenet
sh write_imagenet.sh 256 0.0 100 # raw mode. raw mode donot apply jpg quality
# Writing ImageNet train dataset to /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_0.0_100.ffcv

sh write_imagenet.sh 256 100.0 90 # jpg mode with 90% quality
# Writing ImageNet train dataset to /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_100.0_90.ffcv


⚡ main ~/ffcv-imagenet sh write_imagenet.sh 256 0.0 100
┌ Arguments defined────────┬─────────────────────────────────────────────────────────────────────────────┐
│ Parameter                │ Value                                                                       │
├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ cfg.dataset              │ imagenet                                                                    │
│ cfg.split                │ train                                                                       │
│ cfg.data_dir             │ /teamspace/studios/this_studio/data/imagenet-1m-raw/train                   │
│ cfg.write_path           │ /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_0.0_100.ffcv │
│ cfg.write_mode           │ proportion                                                                  │
│ cfg.max_resolution       │ 256                                                                         │
│ cfg.num_workers          │ 16                                                                          │
│ cfg.chunk_size           │ 100                                                                         │
│ cfg.jpeg_quality         │ 100.0                                                                       │
│ cfg.subset               │ -1                                                                          │
│ cfg.compress_probability │ 0.0                                                                         │
└──────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
100%|███████████████████████████████████████████████████████████████████████████████████| 1281167/1281167 [06:45<00:00, 3158.54it/s]
⚡ main ~/ffcv-imagenet sh write_imagenet.sh 256 100.0 90
Writing ImageNet train dataset to /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_100.0_90.ffcv
┌ Arguments defined────────┬──────────────────────────────────────────────────────────────────────────────┐
│ Parameter                │ Value                                                                        │
├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ cfg.dataset              │ imagenet                                                                     │
│ cfg.split                │ train                                                                        │
│ cfg.data_dir             │ /teamspace/studios/this_studio/data/imagenet-1m-raw/train                    │
│ cfg.write_path           │ /teamspace/studios/this_studio/data/imagenet-1m-ffcv/train_256_100.0_90.ffcv │
│ cfg.write_mode           │ proportion                                                                   │
│ cfg.max_resolution       │ 256                                                                          │
│ cfg.num_workers          │ 16                                                                           │
│ cfg.chunk_size           │ 100                                                                          │
│ cfg.jpeg_quality         │ 90.0                                                                         │
│ cfg.subset               │ -1                                                                           │
│ cfg.compress_probability │ 100.0                                                                        │
└──────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘
100%|███████████████████████████████████████████████████████████████████████████████████| 1281167/1281167 [07:45<00:00, 2753.44it/s]
```

Run prep script for litdata dataset





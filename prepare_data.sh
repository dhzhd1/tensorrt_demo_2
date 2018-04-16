#!/bin/bash

wget "http://download.tensorflow.org/example_images/flower_photos.tgz"
tar xzf flower_photos.tgz
mv flower_photos train
mkdir -p val/daisy
mkdir -p val/dandelion
mkdir -p val/roses
mkdir -p val/sunflowers
mkdir -p val/tulips
find ./train/daisy -name *.jpg | shuf | head -100 | xargs mv -t val/daisy
find ./train/dandelion -name *.jpg | shuf | head -100 | xargs mv -t val/dandelion
find ./train/roses -name *.jpg | shuf | head -100 | xargs mv -t val/roses
find ./train/sunflowers -name *.jpg | shuf | head -100 | xargs mv -t val/sunflowers
find ./train/tulips -name *.jpg | shuf | head -100 | xargs mv -t val/tulips

mkdir -p model/snapshot

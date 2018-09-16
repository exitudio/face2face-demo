#!/bin/bash
F2F="face2face-demo"
P2P="pix2pix-tensorflow"
SOURCE="source"

VIDEO="train_vdo.mov"
NUM_IMAGE=800
NUM_EPOCH=200

# 1. clear old source
rm -rf $SOURCE
mkdir $SOURCE

# 2. Gen images from video to source
cd $SOURCE
python ../$F2F/generate_train_data.py \
  --file ../$VIDEO \
  --num $NUM_IMAGE \
  --landmark-model ../$F2F/shape_predictor_68_face_landmarks.dat
cd ..

# 3. preprocess
#  3.1 Resize original images
python $P2P/tools/process.py \
  --input_dir $SOURCE/original \
  --operation resize \
  --output_dir $SOURCE/original_resized
  
#  3.2 Resize landmark images
python $P2P/tools/process.py \
  --input_dir $SOURCE/landmarks \
  --operation resize \
  --output_dir $SOURCE/landmarks_resized
  
#  3.3 Combine both resized original and landmark images
python $P2P/tools/process.py \
  --input_dir $SOURCE/landmarks_resized \
  --b_dir $SOURCE/original_resized \
  --operation combine \
  --output_dir $SOURCE/combined
  
#  3.4Split into train/val set
python $P2P/tools/split.py \
  --dir $SOURCE/combined
  

# 4. Train the model on the data
python $P2P/pix2pix.py \
  --mode train \
  --output_dir $SOURCE/face2face-model \
  --max_epochs $NUM_EPOCH \
  --input_dir $SOURCE/combined/train \
  --which_direction AtoB
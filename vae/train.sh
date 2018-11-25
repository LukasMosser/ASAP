#!/bin/bash      
make build
trap ctrl_c INT

function ctrl_c() {
        docker stop monitor
        docker rm monitor
}

OF="$PWD/runs/asap_1"
docker run -d --volume=$OF/tensorboard:/opt/app/logs -p 6006:6006 --name monitor asap_monitoring tensorboard --logdir ./logs
docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD/dataset:/opt/app/dataset \
  --volume=$OF/tensorboard:/opt/app/tensorboard \
  --volume=$OF/checkpoints:/opt/app/checkpoints \
  --volume=$OF/output:/opt/app/output \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  asap_train python3 train.py --batch-size 128 --epochs 100 --seed 69 --log-interval 100 --hidden_size 3 --window_size 16 \
--min_inline 1300 --max_inline 1502 --step_inline 2 --min_xline 1500 --max_xline 2002 --step_xline 2 \
--fname_near_stack ./dataset/glitne/flattened/near_64_samples.npy --fname_far_stack ./dataset/glitne/flattened/near_64_samples.npy \
--out_dir output --window_size 16
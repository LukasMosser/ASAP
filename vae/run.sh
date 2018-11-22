docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD:/app \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  asap_train python3 train.py --batch-size 128 --epochs 100 --seed 69 --log-interval 100 --hidden_size 3 --window_size 16 \
--min_inline 1300 --max_inline 1502 --step_inline 2 --min_xline 1500 --max_xline 2002 --step_xline 2 \
--fname_near_stack ./data/glitne/numpy/near_stack.npy  --fname_far_stack ./data/glitne/numpy/far_stack.npy \
--out_dir ./results/model_run1 

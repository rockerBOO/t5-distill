  # --teacher TEACHER
  # --student STUDENT
  # --seed SEED
  # --teacher_device TEACHER_DEVICE
  # --dataset_cache_dir DATASET_CACHE_DIR
  # --batch_size BATCH_SIZE
  # --epochs EPOCHS
  # --steps STEPS
  # --save_every_n_epochs SAVE_EVERY_N_EPOCHS
  # --save_every_n_steps SAVE_EVERY_N_STEPS
  # --validation_split VALIDATION_SPLIT
  # --optimizer_args OPTIMIZER_ARGS
  # --log_with LOG_WITH
teacher="google/t5-efficient-base"
student="google/t5-efficient-tiny"
seed=1234
dataset_cache_dir="/mnt/900/datasets/cache"
batch_size=32
epochs=15
optimizer_args="{'weight_decay': 0.01,'eps':None,'use_orthograd':True,'use_adopt':True}"
log_with="wandb"

accelerate launch main.py --teacher $teacher --student $student --seed $seed --dataset_cache_dir $dataset_cache_dir --batch_size $batch_size --epochs $epochs --optimizer_args "$optimizer_args"

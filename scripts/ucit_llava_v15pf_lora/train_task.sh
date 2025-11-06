TASKID=$1
CONFIG=$2
WORKDIR=$3

if [ "$TASKID" -gt 0 ]; then
    LASTCKPT=$(cat $WORKDIR/task$((TASKID-1))/last_checkpoint)
else
    LASTCKPT=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(config.pretrained_pth)")
fi

echo "######################################################"
echo "############## Train Task $TASKID  ###################"
echo "######################################################"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG"
echo "Work dir: $WORKDIR"
echo "Checkpoint: $LASTCKPT"

# train
torchrun --nproc_per_node=$NGPUS cltuner/tools/train.py $CONFIG  --work-dir $WORKDIR/task$TASKID --deepspeed deepspeed_zero2 --current-task $TASKID --cfg-options model.pretrained_pth=$LASTCKPT --launcher pytorch

# generate answers
CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)
for EVALTASKID in $(seq 0 $TASKID); do
    bash ./scripts/generate.sh $CONFIG $CKPT $EVALTASKID $WORKDIR/results/task$TASKID/task$EVALTASKID 
done

# evalutaion
python cltuner/tools/eval/eval_entry.py $CONFIG --current-task $TASKID --work-dir $WORKDIR


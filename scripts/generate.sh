CONFIG=$1
CKPT=$2
EVALTASKID=$3
OUTPUTDIR=$4

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
echo "####### Generating answers ########"
echo "Config file: $CONFIG"
echo "Checkpoint: $CKPT"
echo "OUTPUTDIR: $OUTPUTDIR"
echo "Eval Task ID: $EVALTASKID"

echo "Using $NGPUS GPUs: ${GPULIST[*]}"

rm -rf "$OUTPUTDIR"
mkdir -p "$OUTPUTDIR"

for IDX in $(seq 0 $((NGPUS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python cltuner/tools/generate.py \
        $CONFIG \
        --num-chunks $NGPUS  \
        --work-dir  $OUTPUTDIR \
        --output-dir  $OUTPUTDIR \
        --current-task $EVALTASKID \
        --chunk-idx $IDX \
        --cfg-options model.pretrained_pth=$CKPT &
done

wait

# merge answers
cat "${OUTPUTDIR}"/chunk*.jsonl > "${OUTPUTDIR}/merge.jsonl"
echo "Merged ${OUTPUTDIR}/chunk*.jsonl to: ${OUTPUTDIR}/merge.jsonl"



source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export NPU_ASD_ENABLE=0
export ACLNN_CACHE_LIMIT=100000
export PYTHONPATH=$PWD:$PYTHONPATH
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6003}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
EXPERIMENT=${EXPERIMENT:-"kinematic_anchor"}
ExpPath=${ExpPath:-"../experiments"}
TENSORBOARD_LOGS_PATH="${ExpPath}/${EXPERIMENT}/tensorboard_dir/"


MM_DATA=${MM_DATA:-"./examples/qwenvla/data_3b_kinematic_anchor.json"}
MM_MODEL=${MM_MODEL:-"./examples/qwenvla/model_3b_anchor_vla.json"}
VOCAB_SIZE=${VOCAB_SIZE:-151936}
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH=${LOAD_PATH:-"../../models/ckpt/mm/Qwen2.5-VL-3B-Instruct_pp1_tp1"}
LOAD_PATH="${ExpPath}/${EXPERIMENT}/ckpt"
SAVE_PATH="${ExpPath}/${EXPERIMENT}/ckpt"

TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
MBS=${MBS:-8}
GRAD_ACC_STEP=${GRAD_ACC_STEP:-8}
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))
TRAIN_ITERS=${TRAIN_ITERS:-60000}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size ${VOCAB_SIZE} \
    --seq-length 1280 \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --lr 2.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --train-iters $TRAIN_ITERS \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed ${SEED} \
    --bf16 \
    --load $LOAD_PATH \
    --use-flash-attn \
    --use-distributed-optimizer \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers $NUM_WORKERS \
    --distributed-timeout-minutes 120 \
"
if [ "$PP" != "1" ]; then
    GPT_ARGS="$GPT_ARGS --variable-seq-lengths "
fi

if [ "$FINETUNE" == "true" ]; then
    GPT_ARGS="${GPT_ARGS} --finetune "
fi

if [ "$RESUME" == "true" ]; then
    GPT_ARGS="${GPT_ARGS} --resume "
fi

echo "GPT_ARGS" $GPT_ARGS

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"
EVAL_ITERS=${EVAL_ITERS:-$TRAIN_ITERS}
EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_ITERS}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters $EVAL_ITERS \
    --save $SAVE_PATH \
    --tensorboard-dir $TENSORBOARD_LOGS_PATH \
    --ckpt-format torch \
"
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p ${ExpPath}/$EXPERIMENT
ANACONDA_DIR=${ANACONDA_DIR:-"$(dirname $(dirname $(which conda)))"}
ENV_NAME=${ENV_NAME:-"py311"}
echo "anaconda_dir" $ANACONDA_DIR
${ANACONDA_DIR}/envs/${ENV_NAME}/bin/python -m torch.distributed.launch \
    $DISTRIBUTED_ARGS \
    --use_env \
    pretrain_anchor_vla.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee ${ExpPath}/${EXPERIMENT}/train_${logfile}.log


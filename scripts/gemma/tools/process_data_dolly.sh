BASE_PATH="/home/luffy/projects/ai/LMOps/minillm"


export TF_CPP_MIN_LOG_LEVEL=3

##DOLLY

# training and validation data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num 1000 \
    --model-type gemma2

# # test data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type gemma2

###SELF INST
# test data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/self-inst/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/self-inst/full \
    --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type gemma2

###SINST
# test data
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/sinst/11_/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/sinst/full \
    --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type gemma2

# ###UINST
# # test data
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/uinst/11_/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/uinst/full \
    --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type gemma2


# ###VICUNA
# # test data
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/vicuna/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/vicuna/full \
    --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type gemma2 



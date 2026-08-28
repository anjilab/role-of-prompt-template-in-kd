BASE_PATH="/home/minillm_ai_safety"

export TF_CPP_MIN_LOG_LEVEL=3

##DOLLY

# training and validation data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path /media/drive2/models/qwen-models/Qwen2.5-7B-Instruct  \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num 1000 \
    --model-type qwen2

# test data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path /media/drive2/models/qwen-models/Qwen2.5-7B-Instruct  \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type qwen2

###SELF INST
# test data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/self-inst/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/self-inst/full \
    --model-path /media/drive2/models/qwen-models/Qwen2.5-7B-Instruct  \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type qwen2

###SINST
# test data
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/sinst/11_/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/sinst/full \
    --model-path /media/drive2/models/qwen-models/Qwen2.5-7B-Instruct  \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type qwen2

# ###UINST
# # test data
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/uinst/11_/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/uinst/full \
    --model-path /media/drive2/models/qwen-models/Qwen2.5-7B-Instruct  \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type qwen2


# ###VICUNA
# # test data
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/vicuna/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/vicuna/full \
    --model-path /media/drive2/models/qwen-models/Qwen2.5-7B-Instruct  \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --test \
    --model-type qwen2 
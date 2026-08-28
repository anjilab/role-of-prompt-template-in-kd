BASE_PATH="/home/minillm_ai_safety"

export TF_CPP_MIN_LOG_LEVEL=3

##DOLLY

# training and validation data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/mix/0.9/gemma2/dolly/full \
    --model-path /media/drive2/models/gemma-models/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num 1000 \
    --model-type gemma2-chat \
    --curriculum-mix-ratio 0.9

# # test data

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/dolly/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/gemma2/dolly/full \
#     --model-path /media/drive2/models/gemma-models/gemma-2-9b-it \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type gemma2-chat \
#     --curriculum-mix-ratio 0.2

##SELF INST
# test data

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/self-inst/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/gemma2/self-inst/full \
#     --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it  \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type gemma2-chat

# ###SINST
# # test data
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/sinst/11_/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/gemma2/sinst/full \
#     --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it  \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type gemma2-chat

# # ###UINST
# # # test data
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/uinst/11_/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/gemma2/uinst/full \
#     --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it  \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type gemma2-chat


# # ###VICUNA
# # # test data
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/vicuna/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/gemma2/vicuna/full \
#     --model-path ${BASE_PATH}/checkpoints/gemma/2/gemma-2-9b-it  \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type gemma2-chat 
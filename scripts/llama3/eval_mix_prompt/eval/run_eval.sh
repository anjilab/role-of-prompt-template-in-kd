base_path="/home/minillm_ai_safety"
port=2040
PEFT_CKPT_NAME=$1
PEFT_CKPT_PATH=$2


if [ "$PEFT_CKPT_NAME" == "llama-3.2-3b-it" -o "$PEFT_CKPT_NAME" == "llama-3.1-8b-it" ]
then
    PEFT_CKPT_NAME="test"
    PEFT_CKPT_PATH="test"
fi

CKPT_NAME=$3
CKPT_PATH=$4

echo "Evaluating PEFT_CKPT_NAME: $PEFT_CKPT_NAME"

for data in dolly self_inst vicuna sinst uinst 
# for data in dolly self_inst vicuna sinst metamath
# for data in dolly self_inst vicuna sinst
# for data in dolly 
do
    # Evaluate SFT
    # for seed in 10 
    for seed in 10 20 30 40 50

    do
        CUDA_VISIBLE_DEVICES=0,1 bash ${base_path}/scripts/llama3/eval_mix_prompt/eval/eval_main_${data}.sh $seed ${PEFT_CKPT_NAME} ${PEFT_CKPT_PATH} ${CKPT_NAME} ${CKPT_PATH}
    done

    # # # Evaluate KD
    # for seed in 10 20 30 40 50
    # do
    #     ckpt="kd/llama-7B-13B-sft"
    #     bash ${base_path}/scripts/llama2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    # done

    # # # Evaluate SeqKD
    # for seed in 10 20 30 40 50
    # do
    #     ckpt="seqkd/llama-7B-13B-sft/"
    #     bash ${base_path}/scripts/llama2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    # done

    # # # Evaluate MiniLLM
    # for seed in 10 20 30 40 50
    # do
    #     ckpt="minillm/7B-init-13B-sft/"
    #     bash ${base_path}/scripts/llama2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    # done
done
fi
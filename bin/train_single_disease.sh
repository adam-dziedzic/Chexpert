timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='single_disease'
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=../
nohup python train_single_disease.py \
--verbose False \
>> train_${DATASET}_${timestamp}.txt 2>&1 &
echo train_${DATASET}_${timestamp}.txt




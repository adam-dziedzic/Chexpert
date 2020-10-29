timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='pleural_effusion'
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=../
nohup python train_pleural_effusion.py \
--verbose False \
>> train_${DATASET}_${timestamp}.txt 2>&1 &
echo train_${DATASET}_${timestamp}.txt



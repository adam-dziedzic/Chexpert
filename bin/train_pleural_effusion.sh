timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='pleural_effusion'
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=../
nohup python train_pleural_effusion.py \
--verbose False \
>> train_${DATASET}_${timestamp}.txt 2>&1 &
echo train_${DATASET}_${timestamp}.txt
[1] 23080
dockuser@nic2:~/code/Chexpert/bin$ echo train_${DATASET}_${timestamp}.txt
train_pleural_effusion_2020-10-29-13-48-58-284364327.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='pleural_effusion'
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=../
nohup python train_pleural_effusion.py \
--verbose False \
>> train_${DATASET}_${timestamp}.txt 2>&1 &
echo train_${DATASET}_${timestamp}.txt




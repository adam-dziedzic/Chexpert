# Cardiomegaly
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='single_disease'
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=../
nohup python train_single_disease.py \
--verbose False \
>> train_${DATASET}_${timestamp}.txt 2>&1 &
echo train_${DATASET}_${timestamp}.txt
[1] 204636
estamp}(venv-tf-py3) dockuser@nic3:~/code/Chexpert/bin$ echo train_${DATASET}_${timestamp}.txt
train_single_disease_2020-10-29-16-17-51-426497739.txt
[1] 206460
(venv-tf-py3) dockuser@nic3:~/code/Chexpert/bin$ echo train_${DATASET}_${timestamp}.txt
train_single_disease_2020-10-29-17-47-53-334858213.txt


# Edema
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='single_disease'
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=../
nohup python train_single_disease.py \
--verbose False \
>> train_${DATASET}_${timestamp}.txt 2>&1 &
echo train_${DATASET}_${timestamp}.txt



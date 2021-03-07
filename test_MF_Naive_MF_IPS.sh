nohup python main.py --model=MF_Naive >MF_Naive.log 2>&1
echo "MF_Naive test finished"
nohup python main.py --model=MF_IPS >MF_IPS.log 2>&1
echo "MF_IPS test finished"
python3.7 run.py --model attunet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model csnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model danet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model dense --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model dlink --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model eanet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model hardalter --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model hardnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model lrfea --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model ocnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model ocrnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model onenet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model pspnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model r2att --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model scseunet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model simple --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model ternaus --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model u2netp --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model cenet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model deeplab --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model lightnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model multires --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model r2 --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model setr --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model u2net --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model unet3p --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model unetpp --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model segnet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model unet --dataset pancreas --mode test --cuda -e 50 --loss ce -r test_result
python3.7 run.py --model unet --dataset pancreas --mode test --cuda -e 50 -b 8 --loss ce -w 0.8 -l 3e-4 --pretrain --checkpoint checkpoints/ssl.pkl -r test_result
python3.7 run.py --model unet --dataset pancreas --mode test --cuda -e 50 -b 8 --loss ce -w 0.8 -l 3e-4 --stn --pretrain --checkpoint checkpoints/ssl.pkl -r test_result 
python3.7 run.py --model unet --dataset pancreas --mode test --cuda -e 50 -b 8 --loss ce -w 0.8 -l 3e-4 --aug --stn --pretrain --checkpoint checkpoints/ssl.pkl -r test_result
python3.7 run.py --model deeplab --dataset pancreas --mode test --cuda -e 50 -b 8 --loss ce -w 0.8 -l 3e-4 --pretrain --checkpoint checkpoints/deeplab-ssl.pkl -r test_result
python3.7 run.py --model deeplab --dataset pancreas --mode test --cuda -e 50 -b 8 --loss ce -w 0.8 -l 3e-4 --stn --pretrain --checkpoint checkpoints/deeplab-ssl.pkl -r test_result 
python3.7 run.py --model deeplab --dataset pancreas --mode test --cuda -e 50 -b 8 --loss ce -w 0.8 -l 3e-4 --aug --stn --pretrain --checkpoint checkpoints/deeplab-ssl.pkl -r test_result
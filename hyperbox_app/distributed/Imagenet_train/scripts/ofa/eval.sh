bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/Imagenet_train/distributed_train.sh 4 \
-c /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/Imagenet_train/scripts/ofa/args.yaml \
--mask /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/Imagenet_train/scripts/ofa/arch.json \
--opt lookahead_rmsproptf --opt-eps .001  --knowledge_distill --kd_ratio 9.0 --teacher_name D-Net-big224 \
--resume /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/Imagenet_train/scripts/ofa/ofa_acc80.46.tar \
--evalonly 1
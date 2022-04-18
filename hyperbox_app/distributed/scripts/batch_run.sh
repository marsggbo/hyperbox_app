
bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/search_nb201.sh search_nb201 2 "trainer.strategy=ddp model.is_net_parallel=False model.optimizer_cfg.lr=0.01" &
sleep 2
bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/search_nb201.sh search_nb201 0,1 "trainer.strategy=ddp model.is_net_parallel=False model.optimizer_cfg.lr=0.02" 
sleep 2 
# bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/search_nb201.sh search_nb201 0,1,2,3 "trainer.strategy=ddp model.is_net_parallel=False model.optimizer_cfg.lr=0.004" 
# sleep 2
# sleep 2 
# bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/search_nb201.sh search_nb201 0 "trainer.strategy=ddp +model.is_net_parallel=True" &
# sleep 2 
# bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/search_nb201.sh search_nb201 1 "trainer.strategy=ddp +model.is_net_parallel=True" &
# sleep 2 
# bash /home/xihe/xinhe/hyperbox_app/hyperbox_app/distributed/search_nb201.sh search_nb201 3 "trainer.strategy=ddp +model.is_net_parallel=True" &

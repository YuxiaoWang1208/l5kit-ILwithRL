FLAG=20220526
#python scripts_/offline_rl.py --imitate_loss_weight 0 --pred_loss_weight 1 --cuda_id 0 --flag $FLAG  &
#python scripts_/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0 --cuda_id 0 --flag $FLAG &
#python scripts_/offline_rl.py --imitate_loss_weight 10 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &
#python scripts_/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &
#python scripts_/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.5 --cuda_id 2 --flag $FLAG
#python scripts_/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.1 --cuda_id 2 --flag $FLAG &
#python scripts_/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.01 --cuda_id 3 --flag $FLAG &
#python scripts_/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.02 --cuda_id 3 --flag $FLAG &



python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &

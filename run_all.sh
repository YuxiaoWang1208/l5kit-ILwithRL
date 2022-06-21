FLAG=2022062000
FLAG_FOR_KILL=1114
#python scripts/offline_rl.py --imitate_loss_weight 0 --pred_loss_weight 1 --cuda_id 0 --flag $FLAG  &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0 --cuda_id 0 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 10 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.5 --cuda_id 2 --flag $FLAG
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.1 --cuda_id 2 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.01 --cuda_id 3 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.02 --cuda_id 3 --flag $FLAG &



python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 2 --no_pretrained --flag $FLAG --flag_for_kill $FLAG_FOR_KILL &
python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 2 --flag $FLAG --flag_for_kill $FLAG_FOR_KILL &

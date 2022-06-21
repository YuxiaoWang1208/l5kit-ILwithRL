FLAG=x20220622
FLAG_FOR_KILL=2114
#python scripts/offline_rl.py --imitate_loss_weight 0 --pred_loss_weight 1 --cuda_id 0 --flag $FLAG  &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0 --cuda_id 0 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 10 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 1 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.5 --cuda_id 2 --flag $FLAG
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.1 --cuda_id 2 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.01 --cuda_id 3 --flag $FLAG &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 0.02 --cuda_id 3 --flag $FLAG &



#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 2 --no_pretrained --start_scene 13 --end_scene 14 --flag $FLAG --flag_for_kill $FLAG_FOR_KILL &
python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 0 --flag $FLAG --start_scene 13 --end_scene 14 --flag_for_kill $FLAG_FOR_KILL &
#python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 2 --no_pretrained --start_scene 0 --end_scene 130 --flag $FLAG --flag_for_kill $FLAG_FOR_KILL &
python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 0 --flag $FLAG --start_scene 0 --end_scene 130 --flag_for_kill $FLAG_FOR_KILL &

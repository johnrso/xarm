MODEL_NAME=0426_ddt_best_variant_shelf_bag
DATA_NAME=0424_place_bag_small_stand_conv_quat

rosrun xarm policy_rollout.py \
  --train_config ${MODEL_DIR}/${MODEL_NAME}/config.yaml \
  --pol_ckpt ${MODEL_DIR}/${MODEL_NAME}/policy_best.ckpt \
  --enc_ckpt ${MODEL_DIR}/${MODEL_NAME}/encoder_best.ckpt \
  --conv_config ${DATASET_DIR}/${DATA_NAME}/config.yaml
  #--traj ${DATASET_DIR}/${DATA_NAME}/train/none/traj_10.h5
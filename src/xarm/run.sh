MODEL_NAME=1231_0430_multiview_all_bag_dinov2/0/
DATA_NAME=0429_place_bag_tj_black_red_conv_multiview_euler


rosrun xarm policy_rollout.py \
  --train_config ${MODEL_DIR}/${MODEL_NAME}/config.yaml \
  --pol_ckpt ${MODEL_DIR}/${MODEL_NAME}/policy_best.ckpt \
  --enc_ckpt ${MODEL_DIR}/${MODEL_NAME}/encoder_best.ckpt \
  --conv_config ${DATASET_DIR}/${DATA_NAME}/config.yaml \
  --tag ${MODEL_NAME}
  # --traj ${DATASET_DIR}/${DATA_NAME}/train/none/traj_10.h5

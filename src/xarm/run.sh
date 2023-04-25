MODEL_NAME=0424_place_bag_40
DATA_NAME=0424_place_bag_small_stand_conv_40

rosrun xarm policy_rollout.py \
  --train_config ${MODEL_DIR}/${MODEL_NAME}/config.yaml \
  --pol_ckpt ${MODEL_DIR}/${MODEL_NAME}/policy_best.ckpt \
  --enc_ckpt ${MODEL_DIR}/${MODEL_NAME}/encoder_best.ckpt \
  --conv_config ${DATASET_DIR}/${DATA_NAME}/config.yaml
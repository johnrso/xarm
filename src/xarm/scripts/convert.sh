python src/demo_to_gdict.py source_dir=${DATASET_DIR}/drawer/pwg_screw_1 vis=True ee_control=True
python src/demo_to_gdict.py source_dir=${DATASET_DIR}/drawer/pwg_screw_2 --config-name config.yaml --config-dir ${DATASET_DIR}/drawer/pwg_screw_1/_conv/multiview_ee_euler &

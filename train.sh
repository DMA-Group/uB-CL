D:/app_daily/minconda3/envs/py39_simcse/python.exe run.py \
  --model_name_or_path "E:/huggingface/hub/sbert-all-mpnet-base-v2" \
  --train_file "E:/Project/EmdForBlock/uBCL/data/all_data.txt" \
  --max_seq_length 128 \
  --grouped_size 10 \
  --pool_type "cls" \
  --use_sparse \
  --loss_type "adaptive" \
  --preprocessing_num_workers 4 \
  --output_dir "./output" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --logging_dir "./logs" \
  --logging_steps 100 \
  --save_steps 500 \
  --evaluation_strategy "steps" \
  --eval_steps 500 \
  --lambda_learning_rate 0.01 \
  --delete_word \
  --delete_word_probability 0.2 \
  --swap_word \
  --swap_word_probability 0.3 \
  --do_train
#  --replace_word True \
#  --replace_word_probability 0.2 \
#  --hnsw_index "hnsw file" \
#  --word2vec_model "word2vec model file"
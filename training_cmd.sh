## Bert 

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best  --word_embedding Bert

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Bert
##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select both  --word_embedding Bert 

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select both --anti_noise --word_embedding Bert 

##
python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best  --word_embedding Bert 

python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Bert 

##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 2 \
     --trainset_spoken_language_select asr_1best  --word_embedding Bert 

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 2 \
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Bert 
##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select asr_1best  --word_embedding Bert

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Bert   

##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select manual_transcript  --word_embedding Bert

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select manual_transcript --anti_noise --word_embedding Bert

## 
python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select both  --word_embedding Bert

python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select both --anti_noise --word_embedding Bert     

## 
python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select manual_transcript  --word_embedding Bert

python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select manual_transcript --anti_noise --word_embedding Bert 


## Word2vec 

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best  --word_embedding Word2vec

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Word2vec
##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select both  --word_embedding Word2vec 

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select both --anti_noise --word_embedding Word2vec 

##
python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best  --word_embedding Word2vec 

python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Word2vec 

##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 2 \
     --trainset_spoken_language_select asr_1best  --word_embedding Word2vec 

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 2 \
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Word2vec 
##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select asr_1best  --word_embedding Word2vec

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select asr_1best --anti_noise --word_embedding Word2vec   

##
python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select manual_transcript  --word_embedding Word2vec

python scripts/slu_baseline.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select manual_transcript --anti_noise --word_embedding Word2vec

## 
python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select both  --word_embedding Word2vec

python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select both --anti_noise --word_embedding Word2vec     

## 
python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select manual_transcript  --word_embedding Word2vec

python scripts/slu_baseline.py --encoder_cell GRU --mlp_num_layers 1 --trainset_augmentation\
     --trainset_spoken_language_select manual_transcript --anti_noise --word_embedding Word2vec 



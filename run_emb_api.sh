set -x

for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i \
    python serve_emb_api.py --host 0.0.0.0 --port $(( 8000 + $i )) &
done





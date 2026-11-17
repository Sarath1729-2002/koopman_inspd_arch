#!/bin/bash
# CPU end-to-end check with a random-init TRM on a 2-augmentation ARC build (~2 min). No GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."
TRM=third_party/TinyRecursiveModels
SP=runs/smoke_tmp
mkdir -p $SP
( cd $TRM && python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi \
    --output-dir ../../$SP/arc1-aug2 --subsets training evaluation concept --test-set-name evaluation --num-aug 2 >/dev/null )
python tests/make_fake_ckpt.py --trm_dir $TRM --data_dir $SP/arc1-aug2 --out $SP/fake/step_0
ARGS="--trm_dir $TRM --data_dir $SP/arc1-aug2 --ckpt $SP/fake/step_0 --out runs/smoke --device cpu --dtype float32 --H_cycles 2 --L_cycles 2 --hidden_size 64 --num_heads 2"
python -m ktrm.rollout $ARGS --n_examples 48 --n_act 6 --batch_size 16 --n_tok 8 --pca_dim 16 --u_dim 8
python -m ktrm.diagnose --out runs/smoke --device cpu --pairs 20000 --mlp_epochs 1 --kae_epochs 1 --horizon 3 > /dev/null
python -m ktrm.explore $ARGS --n_examples 16 --batch_size 16 --rollouts 2 --sigmas 0.1 --n_act 3 --tau 0.9 | tail -8
echo "SMOKE OK (accuracies are 0 by design: random weights)"

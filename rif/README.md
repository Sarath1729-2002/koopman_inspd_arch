# reason_reenc — periodic re-encoding of instructions inside reasoning traces

ReasonIF (ACL 2026) / MathIF: reasoning models break trace-level instructions (<25% compliance) and it worsens
with length. Here: does the instruction decay over the trace (precondition), is it lost (memory) or held but not
applied (control), and does re-encoding it every k tokens fix compliance at no accuracy cost, with the best k
predicted by the measured decay timescale? Instructions: uppercase / no commas / French (rule-checked per
64-token window, no LLM judge). Model: DeepSeek-R1-Distill-Qwen-7B on 200 GSM8K problems. Verdict: rif/report.py.

    cd ~/koopman_inspd_arch && tar xzf reason_reenc.tgz && mkdir -p logs
    module load gcc arrow python/3.11 && source ~/venvs/ouro/bin/activate && export HF_HOME=$SCRATCH/hf
    python -m rif.prep                         # login node: model + data/gsm8k_test.jsonl
    OMP_NUM_THREADS=4 bash tests/smoke_rif.sh  # ~1 min
    sbatch slurm/reason_reenc.sh               # ~20-30 min -> results_rif/REPORT.md

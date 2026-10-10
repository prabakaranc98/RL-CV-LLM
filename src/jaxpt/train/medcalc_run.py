"""One training run on MedCalc-Bench Verified with Tunix: X1 (vanilla GRPO) or X2 (vanilla OPD).

The two share everything (data, rollout, aggregation, optimizer, held-out curve, logging, export) except:
  grpo: reference = the base model, frozen; loss = GRPO (DeepSeekMath), advantage from the group's rewards,
        KL to the reference (k3, beta = 0.04).
  opd:  reference = the teacher (Qwen3-4B); loss = OPD (jaxpt.algorithms.opd), per-token advantage
        log pi_T - log pi_theta. The reward is still computed, for monitoring only.

Outputs in <experiment>/results/<run>/: config.json, completions.jsonl (every completion, train and eval),
tb/ (TensorBoard), hf/ (the final policy as a Hugging Face checkpoint for the E0 vLLM evaluation),
train_summary.json. Live: W&B karan98/rl-cv-llm.
"""
import argparse
import functools
import json
import os
import time

import jax.numpy as jnp
import numpy as np

from ..envs import medcalc
from . import tunix_rl

STUDENT = "Qwen/Qwen3-0.6B-Base"


def parse(algo):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; results go to results/<run>/")
    ap.add_argument("--shots", default="per-calculator")
    ap.add_argument("--input", default="values")
    ap.add_argument("--group", type=int, default=8, help="G, samples per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--hours", type=float, default=2.0, help="wall-clock cap on training (eval included)")
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=0, help="0: full fine-tuning; >0: LoRA on all projections")
    ap.add_argument("--lora-alpha", type=float, default=None, help="default: the rank (scale alpha / rank = 1)")
    if algo == "grpo":
        ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient (k3 estimator)")
        ap.add_argument("--epsilon", type=float, default=0.2)
    else:
        ap.add_argument("--distill-weight", type=float, default=1.0, help="hybrid only: lambda on the teacher term")
        ap.add_argument("--teacher", default="Qwen/Qwen3-4B")
        ap.add_argument("--teacher-size", default="4b", help="Tunix ModelConfig.qwen3_<size>")
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=1024)
    # micro-batches are counted in prompts (each brings G rows)
    ap.add_argument("--rollout-micro", type=int, default=4)
    ap.add_argument("--train-micro", type=int, default=1)
    ap.add_argument("--logps-micro", type=int, default=2)
    # held-out curve: fixed test problems, G samples each at the training temperature -> pass@1 and pass@G
    ap.add_argument("--eval-prompts", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="data order and the held-out set")
    ap.add_argument("--rollout-seed", type=int, default=None,
                    help="sampling seed (default: --seed); change it alone to measure run-to-run variance")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--chunked-logps", dest="chunked_logps", action="store_true", default=True,
                    help="default: per-token log-probs / entropy without [rows, tokens, vocab] logits")
    ap.add_argument("--no-chunked-logps", dest="chunked_logps", action="store_false",
                    help="use Tunix's log-probs instead (X1 run grpo-c used them)")
    ap.add_argument("--export-at-eval", action="store_true",
                    help="save the policy as an HF checkpoint at every held-out eval (greedy-accuracy curve with vLLM)")
    return ap.parse_args()


def fits(examples, prompt_of, tok, limit):
    lens = [len(x) for x in tok([prompt_of(e) for e in examples])["input_ids"]]
    return [e for e, n in zip(examples, lens) if n <= limit]


def eval_problems(test, n, seed):
    """n test problems spread over calculators (round robin over a shuffled list of each calculator's problems)."""
    rng = np.random.default_rng(seed)
    by_calc = {}
    for e in test:
        by_calc.setdefault(e.calculator, []).append(e)
    pools = [list(rng.permutation(v)) for _, v in sorted(by_calc.items())]
    out = []
    while len(out) < n and any(pools):
        for p in pools:
            if p and len(out) < n:
                out.append(p.pop())
    return out


def main(algo, experiment_dir, tag):
    args = parse(algo)
    os.environ["WANDB_MODE"] = "online" if args.wandb else "disabled"     # Tunix always creates a W&B backend
    import optax
    from transformers import AutoTokenizer
    from tunix.rl import rl_cluster as rlc
    from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
    from tunix.rl.rollout.base_rollout import RolloutConfig
    from tunix.sft.metrics_logger import MetricsLoggerOptions
    from ..models.export import export_hf
    from ..rollout.stop_strings import StopStringRollout

    if args.chunked_logps:
        from . import chunked_logps
        chunked_logps.install()        # before anything is traced: policy loss, reference / teacher, eval
    out = tunix_rl.run_dir(experiment_dir, args.run)
    (out / "config.json").write_text(json.dumps({"algo": algo, **vars(args)}, indent=1))
    tok = AutoTokenizer.from_pretrained(STUDENT)
    prompt_of = medcalc.prompter(args.shots, args.input)

    # data: train prompts that fit the sampler's static prompt length; a fixed held-out set from test
    train = medcalc.load("train")
    train_fit = fits(train, prompt_of, tok, args.max_prompt)
    test_fit = fits(medcalc.load("test"), prompt_of, tok, args.max_prompt)
    held_out = eval_problems(test_fit, args.eval_prompts, args.seed)
    examples = train_fit + held_out                        # one index space for the reward lookup
    eval_idx = np.arange(len(train_fit), len(examples))
    print(f"train {len(train_fit)}/{len(train)} prompts <= {args.max_prompt} tokens; held-out {len(held_out)} "
          f"problems over {len({e.calculator for e in held_out})} calculators")
    train_ds = tunix_rl.batches(train_fit, prompt_of, args.prompts_per_step, args.seed, epochs=10,
                                deadline=time.time() + args.hours * 3600)
    eval_ds = [{"prompts": np.array([prompt_of(examples[i]) for i in eval_idx]), "idx": eval_idx}]

    # models: the policy (fp32 master weights, remat) and the frozen reference (base model, or the teacher)
    actor = tunix_rl.load_tunix_qwen3(STUDENT, param_dtype=jnp.float32, remat=True)
    lora_scale = None
    if args.lora_rank:
        from ..models import lora
        alpha = args.lora_alpha or float(args.lora_rank)
        actor = lora.apply_lora(actor, args.lora_rank, alpha)      # Tunix then trains only the LoRA A, B
        lora_scale = alpha / args.lora_rank
    if algo == "grpo":
        reference = tunix_rl.load_tunix_qwen3(STUDENT, param_dtype=jnp.bfloat16)
        algo_cfg = GRPOConfig(num_generations=args.group, num_iterations=1, beta=args.beta,
                              kl_loss_mode="low_var_kl", epsilon=args.epsilon, loss_agg_mode="sequence-mean-token-mean")
    else:   # "opd", or "hybrid" (GRPO's group advantage + the teacher term, per token)
        from ..algorithms import opd  # noqa: F401  (registers the "opd" and "grpo_opd" policy losses)
        reference = tunix_rl.load_tunix_qwen3(args.teacher, param_dtype=jnp.bfloat16, size=args.teacher_size)
        algo_cfg = GRPOConfig(num_generations=args.group, num_iterations=1, beta=1.0,   # beta != 0: teacher log-probs
                              policy_loss_fn="opd" if algo == "opd" else "grpo_opd",
                              loss_agg_mode="sequence-mean-token-mean")
        algo_cfg.distill_weight = args.distill_weight

    rollout = RolloutConfig(max_prompt_length=args.max_prompt, max_tokens_to_generate=args.max_new,
                            kv_cache_size=args.max_prompt + args.max_new, temperature=args.temperature,
                            top_p=1.0, top_k=None,
                            seed=args.seed if args.rollout_seed is None else args.rollout_seed)
    schedule = optax.join_schedules([optax.linear_schedule(0.0, args.lr, args.warmup_steps),
                                     optax.constant_schedule(args.lr)], [args.warmup_steps])
    logging = MetricsLoggerOptions(
        log_dir=str(out / "tb"), project_name="rl-cv-llm", run_name=f"{tag}-{args.run}", flush_every_n_steps=1,
        backend_kwargs={"wandb": {"entity": "karan98", "config": {"algo": algo, **vars(args)}}})
    training = rlc.RLTrainingConfig(
        actor_optimizer=optax.chain(optax.clip_by_global_norm(1.0),
                                    optax.adamw(schedule, b1=0.9, b2=0.99, weight_decay=0.0)),
        eval_every_n_steps=args.eval_every, max_steps=args.max_steps,
        mini_batch_size=args.prompts_per_step,                     # one optimizer update per batch: on-policy
        train_micro_batch_size=args.train_micro, rollout_micro_batch_size=args.rollout_micro,
        compute_logps_micro_batch_size=args.logps_micro, metrics_logging_options=logging)
    m = tunix_rl.mesh()
    tunix_rl.tolerate_structure_mismatch()
    Cluster = tunix_rl.separate_reference_cluster() if algo in ("opd", "hybrid") else rlc.RLCluster
    cluster = Cluster(
        actor=actor, reference=reference, tokenizer=tok,
        cluster_config=rlc.ClusterConfig(
            role_to_mesh={r: m for r in (rlc.Role.ACTOR, rlc.Role.REFERENCE, rlc.Role.ROLLOUT)},
            rollout_engine=functools.partial(StopStringRollout, stop=medcalc.STOP, lora_scale=lora_scale),
            training_config=training, rollout_config=rollout))

    scorer = tunix_rl.MedCalcScorer(examples, tunix_rl.typical_answers(train), args.group,
                                    out / "completions.jsonl", rollout_stats=lambda: cluster.rollout.last)
    scorer.step_of = lambda: cluster.global_steps
    if args.export_at_eval:
        scorer.on_eval_step = lambda step: export_hf(cluster.actor_trainer.model, out / "ckpt" / f"step{step}" / "hf",
                                                     base_repo=STUDENT, lora_scale=lora_scale)
    learner = GRPOLearner(rl_cluster=cluster, algo_config=algo_cfg, reward_fns=[scorer.reward],
                          metric_fns=[scorer.metrics])
    # Tunix logs only kl and pg_clipfrac from the loss's aux by default; entropy is the key collapse signal.
    logged = ["kl", "pg_clipfrac", "entropy", "ppo_kl", "advantage/nonzero_frac", "log_ratio/abs_mean"]
    if algo in ("opd", "hybrid"):
        logged += ["opd/teacher_logp", "opd/student_logp"]
    cluster.actor_trainer.with_rl_metrics_to_log({k: np.mean for k in logged})
    tunix_rl.micro_batch_eval(cluster.actor_trainer, rows=args.train_micro * args.group)

    t0 = time.time()
    learner.train(train_ds, eval_ds=eval_ds)
    train_s = time.time() - t0
    export_hf(cluster.actor_trainer.model, out / "hf", base_repo=STUDENT, lora_scale=lora_scale)
    (out / "train_summary.json").write_text(json.dumps({"train_s": train_s, "steps": int(cluster.global_steps),
                                                         "prompts_seen": int(cluster.global_steps) * args.prompts_per_step}))
    print(f"done: {cluster.global_steps} steps in {train_s / 60:.1f} min; policy exported to {out / 'hf'}")

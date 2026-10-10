"""X3: multi-turn GRPO on MedAgentBench for Qwen3-0.6B (LoRA). Rollouts in vLLM, updates in JAX, one GPU.

Loop, per step:
  1. rollouts: --tasks-per-step training tasks x --group samples at T 1, each a full episode in the official environment
     (jaxpt.envs.medagentbench, identical to the official harness on 1,500 recorded episodes). Generation is token-in /
     token-out through vLLM's completions endpoint: the training sequence is exactly the tokens the policy sampled and
     the tokens it was shown (first prompt = the chat template with thinking off; each environment reply is appended
     as a user turn followed by the same empty-think generation prompt).
  2. reward: Benchmark.shaped (1 for a lookup-gated official pass, partial credit up to 0.4 otherwise); advantage = the
     z-score within each task's group (as in GRPO); groups with equal rewards are skipped.
  3. update: on-policy policy gradient on the LoRA parameters, loss = -A_i * mean over the episode's own tokens of
     log pi, averaged over the step's episodes ("sequence-mean-token-mean"); environment tokens are masked out. One
     AdamW step per batch, so the ratio is 1 and no clipping or reference model is involved (beta = 0).
  4. sync: the merged weights are written to --policy-dir and vLLM reloads them (collective_rpc reload_weights).
Every --eval-every steps: greedy episodes on the held-out patients' official tasks, scored by the official grader and
by the lookup-gated reward.

    python train_agentic_grpo.py --run x3-grpo-lora32 --steps 60
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")      # vLLM holds the rest of the GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import requests  # noqa: E402
from flax import nnx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from jaxpt.envs.medagentbench import Benchmark, Episode  # noqa: E402
from jaxpt.models.export import export_hf  # noqa: E402
from jaxpt.models.lora import apply_lora  # noqa: E402
from jaxpt.train import chunked_logps  # noqa: E402
from jaxpt.train.tunix_rl import load_tunix_qwen3  # noqa: E402

REPO = "Qwen/Qwen3-0.6B"
GEN_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"     # Qwen3 chat template, enable_thinking=False


class Policy:
    """The vLLM server that samples for the current weights."""

    def __init__(self, args):
        self.args, self.url = args, f"http://localhost:{args.port}"

    def start(self):
        vbin = os.path.dirname(self.args.vllm_python)
        env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", PATH=f"{vbin}:{os.environ['PATH']}", VIRTUAL_ENV=os.path.dirname(vbin))
        env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        cmd = [f"{vbin}/vllm", "serve", self.args.policy_dir,
               "--served-model-name", "policy", "--port", str(self.args.port), "--max-model-len", str(self.args.max_len),
               "--gpu-memory-utilization", str(self.args.vllm_memory), "--max-num-seqs", "128", "--seed", "0"]
        self.proc = subprocess.Popen(cmd, env=env, stdout=open(f"{self.args.out}/vllm.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(240):
            try:
                if requests.get(f"{self.url}/v1/models", timeout=5).ok:
                    return
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited; see {self.args.out}/vllm.log")
            time.sleep(2)
        raise TimeoutError("vLLM did not come up")

    def reload(self):
        r = requests.post(f"{self.url}/collective_rpc", json={"method": "reload_weights",
                                                               "kwargs": {"weights_path": self.args.policy_dir}}, timeout=600)
        r.raise_for_status()
        requests.post(f"{self.url}/reset_prefix_cache", timeout=60)

    def sample(self, ids, temperature, seed):
        body = {"model": "policy", "prompt": ids, "max_tokens": self.args.max_tokens, "temperature": temperature,
                "seed": seed, "return_token_ids": True, "logprobs": 0, "skip_special_tokens": True}
        for attempt in range(3):
            try:
                r = requests.post(f"{self.url}/v1/completions", json=body, timeout=600)
                r.raise_for_status()
                c = r.json()["choices"][0]
                lp = (c.get("logprobs") or {}).get("token_logprobs") or []
                return c["text"], list(c["token_ids"]), lp
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)


def run_episode(bench, case, policy, tok, args, temperature, seed):
    """One episode as token ids: (episode, prompt ids, completion ids, mask over completion = sampled tokens,
    vLLM's log-probs of the sampled tokens)."""
    ep = Episode(bench, case, max_rounds=8, max_reply_chars=args.max_reply_chars)
    prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content": ep.history[0].content}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False),
                        add_special_tokens=False)
    seq, mask, vlp = list(prompt), [], []
    while not ep.done:
        if len(seq) + args.max_tokens > args.max_len:
            ep.status = "context limit"
            break
        text, ids, lp = policy.sample(seq, temperature, seed)
        seq += ids
        mask += [1] * len(ids)
        vlp += list(lp) + [None] * (len(ids) - len(lp))
        if not ids or ids[-1] != tok.convert_tokens_to_ids("<|im_end|>"):      # cut by max_tokens: close the turn
            seq.append(tok.convert_tokens_to_ids("<|im_end|>"))
            mask.append(0)
            vlp.append(None)
        ep.step(text)
        if ep.done:
            break
        env = tok.encode(f"\n<|im_start|>user\n{ep.history[-1].content}<|im_end|>\n{GEN_PROMPT}", add_special_tokens=False)
        seq += env
        mask += [0] * len(env)
        vlp += [None] * len(env)
    return ep, prompt, seq[len(prompt):], mask, vlp


C_BUCKETS = (512, 1024)


def replay_demo(bench, case, actions, tok, args):
    """A scripted episode as token ids, in the same format as run_episode: each action's tokens plus <|im_end|> are
    trained (mask 1), the environment's real replies are context (mask 0)."""
    ep = Episode(bench, case, max_rounds=8, max_reply_chars=args.max_reply_chars)
    prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content": ep.history[0].content}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False),
                        add_special_tokens=False)
    seq, mask, truncated = list(prompt), [], False
    for a in actions:
        ids = tok.encode(a, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        seq += ids
        mask += [1] * len(ids)
        ep.step(a)
        if ep.done:
            break
        truncated |= ep.history[-1].content.endswith("...[truncated]")
        env = tok.encode(f"\n<|im_start|>user\n{ep.history[-1].content}<|im_end|>\n{GEN_PROMPT}", add_special_tokens=False)
        seq += env
        mask += [0] * len(env)
    return ep, prompt, seq[len(prompt):], mask, truncated


def bucket(n, sizes=(512, 1024, 1536, 2048, 3072, 4096, 5120, 6144)):
    return next((s for s in sizes if n <= s), n)


def make_steps(pad_id):
    def loss_fn(model, prompt, completion, mask, adv):
        graphdef, state = nnx.split(model)
        logp, ent = chunked_logps.compute_per_token_logps(graphdef, state, prompt, completion, pad_id, -1,
                                                          stop_gradient=False, return_logits=True)
        m = mask.astype(jnp.float32)
        n = jnp.maximum(m.sum(), 1.0)
        return -adv * (logp * m).sum() / n, ((ent * m).sum() / n, jax.lax.stop_gradient(logp))

    @nnx.jit
    def grad_step(model, prompt, completion, mask, adv):
        (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam), has_aux=True)(
            model, prompt, completion, mask, adv)
        return loss, aux, grads

    @nnx.jit
    def apply(optimizer, model, grads, scale):
        grads = jax.tree.map(lambda g: g * scale, grads)
        try:
            optimizer.update(model, grads)
        except TypeError:
            optimizer.update(grads)
    return grad_step, apply


def evaluate(bench, cases, policy, tok, args, step, log):
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(lambda c: run_episode(bench, c, policy, tok, args, 0.0, 0), cases))
    by = collections.defaultdict(lambda: [0, 0, 0])
    for c, (ep, *_rest) in zip(cases, res):
        b = by[c["id"].split("_")[0]]
        b[0] += 1
        b[1] += bench.official(c, ep)
        b[2] += bench.reward(c, ep)
    n = len(cases)
    off, gated = sum(b[1] for b in by.values()), sum(b[2] for b in by.values())
    row = {"eval_step": step, "heldout_official": off / n, "heldout_gated": gated / n, "n": n,
           "statuses": dict(collections.Counter(ep.status for ep, *_ in res)),
           "per_type": {t: b for t, b in sorted(by.items(), key=lambda kv: int(kv[0][4:]))}, "seconds": time.time() - t0}
    log(row)
    print(f"[eval step {step}] held-out official {100 * off / n:.1f}% ({off}/{n})  gated {100 * gated / n:.1f}% ({gated}/{n})"
          f"  {row['statuses']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mab", default="/root/MedAgentBench")
    ap.add_argument("--train", default="/root/mab_train/train_official.json")
    ap.add_argument("--heldout", default="/root/mab_train/heldout_official.json")
    ap.add_argument("--out-root", default="/root/x3_runs")
    ap.add_argument("--policy-dir", default="/root/x3_policy")
    ap.add_argument("--vllm-python", default="/root/venv-vllm/bin/python")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--vllm-memory", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tasks-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256, help="per turn")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-reply-chars", type=int, default=3000)
    ap.add_argument("--train-max-len", type=int, default=4608,
                    help="tokens of an episode that enter the gradient: the prompt (3,584 bucket) + the first 1,024. "
                         "Plain attention memory grows with length squared: 6,144 and 9,728 tokens ran out of the "
                         "A5000's JAX share (6.8 and 8.0 GiB single allocations). Later tokens are dropped.")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=None, help="stop RL after this long, then run the final eval")
    ap.add_argument("--sft-demos", default=None, help="warm-up: build_demos.py output; SFT on them before RL")
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--wandb", action="store_true", help="log to W&B project rl-cv-llm as x3-<run>")
    args = ap.parse_args()
    args.out = f"{args.out_root}/{args.run}"
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    metrics = open(f"{args.out}/metrics.jsonl", "a")
    trajectories = open(f"{args.out}/trajectories.jsonl", "a")
    wb = None

    def log(row):
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if wb is not None:
            flat = {k: v for k, v in row.items() if isinstance(v, (int, float)) and v is not None}
            if "eval_step" in row:
                wb.log({f"heldout/{k}": v for k, v in flat.items() if k != "eval_step"}, step=row["eval_step"])
            elif "step" in row:
                wb.log({f"train/{k}": v for k, v in flat.items() if k != "step"}, step=row["step"])
            # warm-up (SFT) rows stay in metrics.jsonl: they have no RL step

    bench = Benchmark(args.mab)
    train, heldout = json.load(open(args.train)), json.load(open(args.heldout))
    tok = AutoTokenizer.from_pretrained(REPO)
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")

    chunked_logps.install()
    model = load_tunix_qwen3(REPO, param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, size="0p6b", remat=True)
    model = apply_lora(model, args.lora_rank, args.lora_alpha)
    scale = args.lora_alpha / args.lora_rank
    sched = optax.join_schedules([optax.linear_schedule(args.lr / args.warmup_steps, args.lr, args.warmup_steps),
                                  optax.constant_schedule(args.lr)], [args.warmup_steps])
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.0)),
                              wrt=nnx.LoRAParam)
    grad_step, apply = make_steps(pad_id)

    # preflight: the largest gradient shape must fit before anything else starts
    P, C = 3584, C_BUCKETS[-1]
    t0 = time.time()
    _, _, g = grad_step(model, jnp.full((1, P), 100, jnp.int32), jnp.full((1, C), 100, jnp.int32),
                        jnp.ones((1, C), jnp.int32), jnp.float32(0.0))
    jax.block_until_ready(g)
    stats = jax.devices()[0].memory_stats() or {}
    print(f"preflight ok: grad step at {P}+{C} tokens compiled in {time.time() - t0:.0f}s, peak "
          f"{stats.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB of {stats.get('bytes_limit', 0) / 2**30:.1f} GiB", flush=True)
    del g
    if args.wandb:
        import wandb
        wb = wandb.init(project="rl-cv-llm", name=f"x3-{args.run}", config=vars(args))

    if args.sft_demos:
        demos = [json.loads(l) for l in open(args.sft_demos)]
        keep, why = [], collections.Counter()
        for d in demos:
            ep, prompt, completion, mask, truncated = replay_demo(bench, d["case"], d["actions"], tok, args)
            answer = json.loads(ep.result) if ep.result else None
            if bench.reward(d["case"], ep) != 1.0:
                why["not a gated pass"] += 1
            elif truncated and answer not in ([-1], []):
                why["reply truncated: answer may be hidden"] += 1
            elif len(completion) > args.train_max_len - 3584:
                why["answer beyond the trained window"] += 1
            else:
                keep.append((d["case"]["id"], prompt, completion, mask))
        types = collections.Counter(k[0].split("_")[0] for k in keep)
        print(f"SFT demos kept {len(keep)} / {len(demos)}; dropped {dict(why)}; per type {dict(types)}", flush=True)
        log({"sft_demos_kept": len(keep), "sft_demos_total": len(demos), "sft_dropped": dict(why), "sft_per_type": dict(types)})
        sft_opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.sft_lr, weight_decay=0.0)),
                                wrt=nnx.LoRAParam)
        rng_sft = random.Random(args.seed)
        t0 = time.time()
        for epoch in range(args.sft_epochs):
            rng_sft.shuffle(keep)
            for b in range(0, len(keep), args.sft_batch):
                acc, nll = None, []
                for _, prompt, completion, mask in keep[b:b + args.sft_batch]:
                    p = np.full((1, 3584), pad_id, np.int32)
                    p[0, 3584 - len(prompt):] = prompt
                    C = bucket(len(completion), C_BUCKETS)
                    c = np.full((1, C), pad_id, np.int32)
                    c[0, :len(completion)] = completion
                    m = np.zeros((1, C), np.int32)
                    m[0, :len(mask)] = mask
                    loss, _, g = grad_step(model, jnp.asarray(p), jnp.asarray(c), jnp.asarray(m), jnp.float32(1.0))
                    acc = g if acc is None else jax.tree.map(jnp.add, acc, g)
                    nll.append(float(loss))
                apply(sft_opt, model, acc, jnp.float32(1.0 / len(nll)))
                row = {"sft_epoch": epoch, "sft_batch": b // args.sft_batch, "nll": float(np.mean(nll)),
                       "seconds": time.time() - t0}
                log(row)
                print(f"sft epoch {epoch} batch {b // args.sft_batch:3d} nll {row['nll']:.4f} ({row['seconds']:.0f}s)", flush=True)
        del sft_opt

    export_hf(model, args.policy_dir, base_repo=REPO, lora_scale=scale)          # step 0: the base model, or after SFT
    policy = Policy(args)
    policy.start()
    import atexit
    atexit.register(lambda: policy.proc.poll() is None and policy.proc.terminate())
    print(f"vLLM up; JAX {jax.devices()}", flush=True)

    rng = random.Random(args.seed)
    order = []
    evaluate(bench, heldout, policy, tok, args, 0, log)
    t_start = time.time()
    for step in range(1, args.steps + 1):
        t0 = time.time()
        if len(order) < args.tasks_per_step:
            more = list(train)
            rng.shuffle(more)
            order += more
        tasks, order = order[: args.tasks_per_step], order[args.tasks_per_step:]
        jobs = [(c, g) for c in tasks for g in range(args.group)]
        with ThreadPoolExecutor(args.workers) as pool:
            eps = list(pool.map(lambda j: run_episode(bench, j[0], policy, tok, args, args.temperature,
                                                      zlib.crc32(f"{args.seed}/{step}/{j[0]['id']}/{j[1]}".encode()) % (1 << 31)), jobs))
        t_roll = time.time() - t0

        rewards = np.array([bench.shaped(c, e[0]) for (c, _), e in zip(jobs, eps)], np.float32)
        r = rewards.reshape(len(tasks), args.group)
        adv = ((r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-6)).reshape(-1)
        adv[np.repeat(r.std(1) == 0, args.group)] = 0.0

        grads_acc, ent_sum, lp_gap, n_train, tokens, kept_tokens, all_tokens = None, 0.0, [], 0, 0, 0, 0
        t1 = time.time()
        for i in np.argsort([-len(e[2]) for e in eps]):
            if adv[i] == 0.0:
                continue
            ep, prompt, completion, mask, vlp = eps[i]
            keep = max(args.train_max_len - bucket(len(prompt), (3584, 4096)), 0)
            kept_tokens += sum(mask[:keep])
            all_tokens += sum(mask)
            completion, mask, vlp = completion[:keep], mask[:keep], vlp[:keep]
            if not any(mask):
                continue
            P, C = bucket(len(prompt), (3584, 4096)), bucket(len(completion), C_BUCKETS)
            p = np.full((1, P), pad_id, np.int32)
            p[0, P - len(prompt):] = prompt
            c = np.full((1, C), pad_id, np.int32)
            c[0, :len(completion)] = completion
            m = np.zeros((1, C), np.int32)
            m[0, :len(mask)] = mask
            loss, (ent, logp), grads = grad_step(model, jnp.asarray(p), jnp.asarray(c), jnp.asarray(m),
                                                 jnp.float32(adv[i]))
            grads_acc = grads if grads_acc is None else jax.tree.map(jnp.add, grads_acc, grads)
            ent_sum += float(ent)
            n_train += 1
            tokens += len(prompt) + len(completion)
            if len(lp_gap) < 4:                                   # sampler vs trainer log-prob agreement
                lpj = np.asarray(logp)[0, :len(completion)]
                pairs = [(a, b) for a, b, k in zip(vlp, lpj, mask) if k and a is not None]
                if pairs:
                    lp_gap.append(float(np.mean([abs(a - b) for a, b in pairs])))
        if grads_acc is not None:
            apply(optimizer, model, grads_acc, jnp.float32(1.0 / len(eps)))
        t_train = time.time() - t1

        t2 = time.time()
        if grads_acc is not None:
            export_hf(model, args.policy_dir, base_repo=REPO, lora_scale=scale)
            policy.reload()
        t_sync = time.time() - t2

        statuses = collections.Counter(e[0].status for e in eps)
        for i in sorted(range(len(eps)), key=lambda i: -rewards[i])[:2] + [int(np.argmin(rewards))]:
            (c, g), ep = jobs[i], eps[i][0]
            trajectories.write(json.dumps({"step": step, "id": c["id"], "sample": g, "reward": float(rewards[i]),
                                           "advantage": float(adv[i]), "status": ep.status, "answer": ep.result,
                                           "turns": [t.content[:600] for t in ep.history[1:]]}) + "\n")
        trajectories.flush()
        row = {"step": step, "reward_mean": float(rewards.mean()),
               "official_rate": float(np.mean([bench.official(c, e[0]) for (c, _), e in zip(jobs, eps)])),
               "gated_rate": float(np.mean([bench.reward(c, e[0]) for (c, _), e in zip(jobs, eps)])),
               "lookup_rate": float(np.mean([bench.looked_up(c, e[0]) and c["id"].split("_")[0] in
                                             ("task1", "task2", "task4", "task5", "task6", "task7", "task9", "task10")
                                             for (c, _), e in zip(jobs, eps)])),
               "signal_groups": int((r.std(1) > 0).sum()), "groups": len(tasks), "trained_episodes": n_train,
               "tokens": tokens, "entropy": ent_sum / max(n_train, 1),
               "policy_token_frac_trained": kept_tokens / max(all_tokens, 1),
               "sampler_trainer_logp_gap": float(np.mean(lp_gap)) if lp_gap else None,
               "statuses": dict(statuses), "t_rollout": t_roll, "t_train": t_train, "t_sync": t_sync}
        log(row)
        print(f"step {step:3d} reward {row['reward_mean']:.3f} official {row['official_rate']:.3f} gated "
              f"{row['gated_rate']:.3f} lookup {row['lookup_rate']:.3f} signal {row['signal_groups']}/{len(tasks)} "
              f"trained {n_train} ent {row['entropy']:.3f} lpgap {row['sampler_trainer_logp_gap']} "
              f"| roll {t_roll:.0f}s train {t_train:.0f}s sync {t_sync:.0f}s", flush=True)
        out_of_time = args.hours is not None and time.time() - t_start > args.hours * 3600
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            evaluate(bench, heldout, policy, tok, args, step, log)
            shutil.copy(f"{args.policy_dir}/model.safetensors", f"{args.out}/model_step{step}.safetensors")
        if out_of_time:
            print(f"time limit reached after step {step}", flush=True)
            break
    policy.proc.terminate()
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()

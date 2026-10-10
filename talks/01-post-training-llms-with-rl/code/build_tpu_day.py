"""Assemble the 30-slide TPU-day deck (tpu-day.md) from the full deck and the new slides in tpu-day-new.md.

Slides are taken from slides.md (with its <!-- include --> parts spliced in) by their exact title, so the short deck
stays in sync with the full one: fix a slide there, rerun this. New slides live in tpu-day-new.md, one per
<!-- @key --> marker. Run from the deck directory:  python3 code/build_tpu_day.py && DECK=tpu-day ./build.sh deck
"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parents[1]

ORDER = [
    ("new", "title"), ("new", "outcomes"),
    ("old", "Post-training turns a next-token predictor into a useful model"),
    ("new", "mindmap"), ("new", "timeline"),
    ("divider", "Part 1|The ideas, in plain words|Reinforcement learning, a language model as a policy, and four ways to train it."),
    ("old", "An agent learns from the consequences of its own actions"),
    ("old", "The model maps a context to a distribution over the next token"),
    ("old", "Generating a response is a sequence of decisions"),
    ("old", "The policy gradient: make good actions more likely"),
    ("old", "RLHF: learn a reward from people, then optimize it with PPO"),
    ("old", "Optimized too hard, the policy exploits the reward model"),
    ("old", "RL with verifiable rewards: a program checks the answer"),
    ("old", "GRPO on one prompt"),
    ("old", "The GRPO advantage is the reward, standardized within its group"),
    ("old", "Three ways to learn from a teacher"),
    ("old", "On-policy distillation: the teacher scores every token"),
    ("old", "Four methods, one loop"),
    ("old", "Generation dominates the time: a rollout engine and a trainer"),
    ("old", "JAX: composable transformations of pure functions"),
    ("old", "A GRPO update in JAX"),
    ("new", "recipe"),
    ("divider", "Part 2|Two experiments on a 0.6B model|Clinical calculation (one checked answer) and an agent in a health record (many turns)."),
    ("old", "The experiments: two benchmarks, GRPO and OPD, everything measured"),
    ("old", "The task: clinical calculators, with answers a program can check"),
    ("old", "One loop for both methods: only the per-token signal differs"),
    ("old", "With LoRA on both, OPD keeps climbing where GRPO flattens"),
    ("old", "On the full test, OPD gains about 15 points and GRPO 6"),
    ("old", "All four conditions: OPD transfers to reading the note; nobody learns the formulas"),
    ("new", "glossary"),
    ("old", "How agentic RL runs: vLLM plays the episodes, JAX learns from them"),
    ("old", "One episode, token by token: what the model writes is all that is trained"),
    ("old", "The 0.6B's 6% is a guess, and RL has nothing to learn from"),
    ("old", "Run 1: GRPO learned the lookup, then learned to never answer"),
    ("old", "Warm start, then RL: real success 0% → 39% → 47%"),
    ("old", "Three runs, one picture: RL needs a model that sometimes succeeds"),
    ("new", "takeaways"), ("new", "questions"),
]


def full_deck():
    text = (HERE / "slides.md").read_text()
    for name in re.findall(r"<!-- include: ([\w.-]+\.md) -->", text):
        text = text.replace(f"<!-- include: {name} -->", (HERE / name).read_text().rstrip() + "\n")
    front, body = text.split("\n---\n", 1)[0] + "\n---\n", text.split("\n---\n", 1)[1]
    return front, [b.strip("\n") for b in re.split(r"\n---\n", body)]


def title_of(block):
    for line in block.splitlines():
        m = re.match(r"#{1,6} (.+)", line)
        if m:
            return m.group(1).strip()
    return None


def main():
    front, blocks = full_deck()
    by_title = {}
    for b in blocks:
        t = title_of(b)
        if t and t not in by_title:
            by_title[t] = b
    new = dict(re.findall(r"<!-- @(\w+) -->\n(.*?)(?=\n<!-- @\w+ -->|\Z)", (HERE / "tpu-day-new.md").read_text(), re.S))
    out, missing = [], []
    for kind, key in ORDER:
        if kind == "new":
            out.append(new[key].strip("\n"))
        elif kind == "divider":
            part, title, sub = key.split("|")
            out.append(f"<!-- _class: divider -->\n\n###### {part}\n\n# {title}\n\n{sub}")
        elif key in by_title:
            out.append(by_title[key])
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"slides not found in the full deck: {missing}")
    front = front.replace('title: "Post-training LLMs with RL and On-Policy Distillation"',
                          'title: "Post-training LLMs with RL and On-Policy Distillation (TPU day)"')
    (HERE / "tpu-day.md").write_text(front + "\n" + "\n\n---\n\n".join(out) + "\n")
    print(f"tpu-day.md: {len(out)} slides")


if __name__ == "__main__":
    main()

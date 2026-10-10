"""MedAgentBench as an RL environment: the official protocol and grader, plus a lookup-gated training reward.

Mirrors stanfordmlgroup/MedAgentBench src/server/tasks/medagentbench/__init__.py turn for turn: the same prompt text,
the same action parsing (GET / POST / FINISH, anything else ends the episode as an invalid action), the same replies
from the environment, and the same 8-round limit. POSTs are never sent to the server (the official harness only checks
that the payload is JSON); the grader reads them from the history. tests: experiments/x3_mab_agentic/replay_check.py
replays recorded official episodes through this class and must reproduce every reply and every grade.

Rewards:
  official(case, ep)  the benchmark's own grade (refsol.py), the number every result is reported in;
  reward(case, ep)    for training only: the official grade, and 0 unless the agent made the lookup the task needs
                      (a successful GET of that patient's record or lab code). Without the gate, FINISH([]) with no
                      action already passes 70 / 300 official tasks, and GRPO learns that first.

Training may cap each environment reply (max_reply_chars); the official harness never truncates, so evaluation does not.
"""
import importlib
import json
import os
import sys
import threading

PROMPT = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""

FINISH_HINT = "Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
# the lookup each task type needs: (resource, query fragment); MRN-based tasks must also name the patient
LOOKUP = {"task1": ("Patient?", None), "task2": ("Patient?", None), "task4": ("Observation?", "code=MG"),
          "task5": ("Observation?", "code=MG"), "task6": ("Observation?", "code=GLU"),
          "task7": ("Observation?", "code=GLU"), "task9": ("Observation?", "code=K"),
          "task10": ("Observation?", "code=A1C")}


class Turn:
    """One history item as refsol.py reads it (AgentBench's ChatHistoryItem: role is "user" or "agent")."""
    def __init__(self, role, content):
        self.role, self.content = role, content


class Result:
    def __init__(self, result, history):
        self.result, self.history = result, history


class Benchmark:
    """The official pieces, loaded from a MedAgentBench checkout: task files, function list, grader, FHIR base URL."""

    def __init__(self, root="/root/MedAgentBench", fhir_base=None):
        sys.path.insert(0, root)
        cwd = os.getcwd()
        os.chdir(root)
        try:
            self.refsol = importlib.import_module("src.server.tasks.medagentbench.refsol")
        finally:
            os.chdir(cwd)
        self.root = root
        self.funcs = json.load(open(f"{root}/data/medagentbench/funcs_v1.json"))
        cfg = open(f"{root}/configs/tasks/medagentbench.yaml").read()
        self.base = fhir_base or cfg.split('fhir_api_base: "', 1)[1].split('"', 1)[0]
        self._get, self._cache, self._lock = self.refsol.send_get_request, {}, threading.Lock()
        self.refsol.send_get_request = self.get          # the grader shares the cache
        self.refsol.print = lambda *a, **k: None        # silence its per-task prints

    def tasks(self, name="test_data_v2.json"):
        return json.load(open(f"{self.root}/data/medagentbench/{name}"))

    def get(self, url, params=None, headers=None):
        """The harness's GET (same return format), cached: the server is read-only, since POSTs are never sent."""
        with self._lock:
            if url in self._cache:
                return self._cache[url]
        r = self._get(url, params, headers)
        if "data" in r:
            with self._lock:
                self._cache[url] = r
        return r

    def prompt(self, case):
        return PROMPT.format(api_base=self.base, functions=json.dumps(self.funcs), context=case["context"],
                             question=case["instruction"])

    def official(self, case, ep):
        """The benchmark's grade: only finished episodes are graded (eval.py: any grader exception counts as wrong)."""
        if ep.result is None:
            return False
        try:
            return getattr(self.refsol, case["id"].split("_")[0])(case, Result(ep.result, ep.history), self.base) is True
        except Exception:
            return False

    def looked_up(self, case, ep):
        task = case["id"].split("_")[0]
        if task not in LOOKUP:
            return True
        resource, fragment = LOOKUP[task]
        mrn = case.get("eval_MRN")
        for turn, reply in zip(ep.history[1::2], ep.history[2::2]):
            url = turn.content.strip()
            if (url.startswith("GET") and resource in url and (fragment is None or fragment in url)
                    and (task == "task1" or mrn in url) and reply.content.startswith("Here is the response")):
                return True
        return False

    def reward(self, case, ep):
        return float(self.official(case, ep) and self.looked_up(case, ep))

    def shaped(self, case, ep):
        """Training reward for a policy that never passes yet (Qwen3-0.6B: 0 / 1,664 samples at T 1). 1.0 for a gated
        pass; otherwise partial credit, at most 0.4, for the two halves it gets right separately and their
        combination: an action on the real server URL (+0.1, it tends to write the {api_base} placeholder), the
        right request whatever the URL (+0.1: the task's lookup, or the POST resource for task3 / task8), and the
        task's lookup succeeding (+0.2). Partial credit only for an episode that ends with FINISH: without that
        condition the first run (grpo-shaped-lora32-t4k) learned to keep issuing GETs until the round limit,
        collecting the 0.4 and never answering (finished episodes 61 / 64 at step 1, 0 / 64 by step 9; held-out
        official 19.6% -> 0%)."""
        if self.reward(case, ep):
            return 1.0
        if ep.status != "completed":
            return 0.0
        task, mrn = case["id"].split("_")[0], case.get("eval_MRN") or ""
        actions = [t.content.strip() for t in ep.history[1::2]]
        base_ok = any(a.split("\n")[0][4:].strip().startswith(self.base) for a in actions if a[:4] in ("GET ", "POST"))
        if task in LOOKUP:
            resource, fragment = LOOKUP[task]
            query_ok = any(a.startswith("GET") and resource in a and (fragment is None or fragment in a)
                           and (task == "task1" or mrn in a) for a in actions)
            lookup_ok = self.looked_up(case, ep)
        else:
            resource = {"task3": "Observation", "task8": "ServiceRequest"}[task]
            query_ok = any(a.startswith("POST") and resource in a.split("\n")[0] for a in actions)
            lookup_ok = False
        return 0.1 * base_ok + 0.1 * query_ok + 0.2 * lookup_ok


class Episode:
    """One rollout of one task: step(agent_text) returns the environment's reply, or None once the episode is over."""

    def __init__(self, bench, case, max_rounds=8, max_reply_chars=None):
        self.bench, self.case, self.max_rounds, self.max_reply_chars = bench, case, max_rounds, max_reply_chars
        self.history = [Turn("user", bench.prompt(case))]
        self.rounds, self.status, self.result = 0, None, None

    @property
    def done(self):
        return self.status is not None

    def _reply(self, text):
        if self.max_reply_chars is not None and len(text) > self.max_reply_chars:
            text = text[: self.max_reply_chars] + " ...[truncated]"
        self.history.append(Turn("user", text))
        return text

    def step(self, text):
        assert not self.done
        self.history.append(Turn("agent", text))
        self.rounds += 1
        r = text.strip().replace("```tool_code", "").replace("```", "").strip()
        if r.startswith("GET"):
            res = self.bench.get(r[3:].strip() + "&_format=json")
            reply = (f"Here is the response from the GET request:\n{res['data']}. {FINISH_HINT}" if "data" in res
                     else f"Error in sending the GET request: {res['error']}")
        elif r.startswith("POST"):
            try:
                json.loads("\n".join(r.split("\n")[1:]))
            except Exception:
                reply = "Invalid POST request"
            else:
                reply = f"POST request accepted and executed successfully. {FINISH_HINT}"
        elif r.startswith("FINISH("):
            self.status, self.result = "completed", r[len("FINISH("):-1]
            return None
        else:
            self.status = "agent invalid action"
            return None
        if self.rounds >= self.max_rounds:
            self._reply(reply)
            self.status = "task limit reached"
            return None
        return self._reply(reply)

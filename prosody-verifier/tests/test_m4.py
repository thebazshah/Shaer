"""M4: the hemistich search, best-of-N selection and the generator's cache reuse.

The generator tests use a tiny random Llama (no download) and a character tokenizer, and run only where torch and
transformers are installed (the notebook's environment); the search and selection tests need only the verifier.
Run: python -m pytest prosody-verifier/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m4_search import (Candidate, Check, CheckerPool, PoemChecker, PoemState, SearchRunner, SearchVerifier,  # noqa: E402
                       best_of_n, defects, poem_rank)
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.verify import Verifier  # noqa: E402

LEX = Lexicon.load(str(HERE / "data" / "lexicon_train.json.gz"))

# a human poem in al-kāmil (tām), rawī د (from the M1 textbook cases), and lines of another rhyme
KAMIL = ["يا أَيُّها الرَجُلُ المُعَلِّمُ غَيرَهُ", "هَلّا لِنَفسِكَ كانَ ذا التَعليمُ",
         "اِبدَأ بِنَفسِكَ فَاِنهَها عَن غَيِّها", "فَإِذا اِنتَهَت عَنهُ فَأَنتَ حَكيمُ"]
FOREIGN = "قِفا نَبكِ مِن ذِكرى حَبيبٍ وَمَنزِلِ"          # al-ṭawīl, rawī ل


@pytest.fixture(scope="module")
def checker():
    return PoemChecker(SearchVerifier(LEX))


# ---------------------------------------------------------------------------------------------- the checker
def test_true_lines_pass_and_a_foreign_rhyme_fails(checker):
    for i, line in enumerate(KAMIL):
        assert checker.check(KAMIL[:i], line, "kamil", "tam").ok, (i, line)
    bad = checker.check(KAMIL[:3], "وَإِذا مَشى في القَومِ كانَ الأَوحَدا", "kamil", "tam")
    assert not bad.ok and bad.new >= 1 and "rawi" in bad.detail


def test_binding_mode_checker():
    pool = CheckerPool(workers=0, binding=True)
    hit, checks = pool.run([(0, KAMIL[:3], ["وَإِذا مَشى في القَومِ كانَ الأَوحَدا", KAMIL[3]], "kamil", "tam")])[0]
    assert hit == 1 and not checks[0].ok
    assert pool.local.v.ignore_diacritics is False


def test_wrong_meter_and_empty_candidates_fail(checker):
    assert not checker.check([], FOREIGN, "kamil", "tam").ok          # a ṭawīl line is not a kāmil sadr
    for lines in ([], KAMIL[:1]):
        c = checker.check(lines, "  ", "kamil", "tam")
        assert not c.ok and c.cost == float("inf")


def test_repeated_rhyme_word_is_a_defect_only_when_asked():
    pv = SimpleNamespace(bayts=[SimpleNamespace(index=0, ok=True)], issues=[{"kind": "count", "bayts": []}],
                         rhyme={"issues": [{"kind": "ita", "bayts": [3]}, {"kind": "rawi", "bayts": [2]}]})
    assert defects(pv, ita=True) == {("ita", 3), ("rawi", 2)}
    assert defects(pv, ita=False) == {("rawi", 2)}


def test_a_rhyme_word_repeated_in_the_second_bayt_is_rejected(checker):
    """Two bayts ending in the same word read as a refrain (radīf) poem: the earlier rhyme words must then rhyme and
    the first bayt lacks the refrain, so the candidate adds defects either way."""
    again = "وَإِذا اِنتَهَيتَ فَأَنتَ ثَمَّ حَكيمُ"
    c = checker.check(KAMIL[:2] + [KAMIL[2]], "فَإِذا اِنتَهَت عَنهُ فَأَنتَ التَعليمُ", "kamil", "tam")
    assert not c.ok and c.new >= 1


def test_search_verifier_agrees_with_the_evaluation_verifier_without_tadwir():
    ev = Verifier(LEX, ignore_diacritics=True, diagnose=False)
    sv = SearchVerifier(LEX)
    for lines in (KAMIL, KAMIL[:2] + [KAMIL[2], "وَإِذا مَشى في القَومِ كانَ الأَوحَدا"], KAMIL[:3] + [FOREIGN[:-6]]):
        a, b = ev.poem(lines, "kamil", "tam"), sv.poem(lines, "kamil", "tam")
        assert [x.ok for x in a.bayts] == [x.ok for x in b.bayts]
        assert a.rhyme["ok"] == b.rhyme["ok"] and defects(a) == defects(b)


def test_memoised_bayts_are_not_shared_between_positions():
    sv = SearchVerifier(LEX)
    pv = sv.poem(KAMIL[:2] * 2, "kamil", "tam")                       # the same bayt twice
    assert [b.index for b in pv.bayts] == [0, 1]
    assert ("ita", 1) in defects(pv)


# ---------------------------------------------------------------------------------------------- the runner
class ListGenerator:
    """Proposes scripted candidates: script[key][line index] is a list of texts, served k at a time."""

    def __init__(self, script):
        self.script, self.served = script, {}

    def propose(self, states, k):
        props = []
        for s in states:
            key = (s.key, len(s.lines))
            start = self.served.get(key, 0)
            texts = (self.script[s.key][len(s.lines)] * 10)[start:start + k]
            self.served[key] = start + k
            props.append([Candidate(t, [len(s.lines)] * (len(t.split()) + 1), len(t.split()) + 1,
                                    eos=t.endswith("<eos>")) for t in texts])
        return props, {"seconds": [0.01] * len(states), "prefill_tokens": [len(s.context) for s in states]}


@pytest.fixture(scope="module")
def pool():
    return CheckerPool(workers=0)


def _state(key="p", n=4):
    return PoemState(key, "kamil", "tam", n, [0] * 5, 5)


def test_runner_takes_the_first_passing_candidate_and_counts_every_token(pool):
    script = {"p": [[FOREIGN, KAMIL[0], KAMIL[0]], [KAMIL[1]], [KAMIL[2]], [FOREIGN, FOREIGN, KAMIL[3]]]}
    s = _state()
    SearchRunner(ListGenerator(script), pool, k=3, rounds=1).run([s])
    assert s.lines == KAMIL and not s.forced
    assert [c for c, _, _ in s.log] == [2, 1, 1, 3]                     # checks stop at the first pass
    assert s.candidates == 12                                            # but all k candidates were generated
    assert s.gen_tokens == sum(len(t.split()) + 1 for t in [FOREIGN, KAMIL[0], KAMIL[0]] + [KAMIL[1]] * 3
                               + [KAMIL[2]] * 3 + [FOREIGN, FOREIGN, KAMIL[3]])
    steps = [len(t.split()) + 1 for t in KAMIL]
    assert s.prefill_tokens == sum(5 + sum(steps[:j]) for j in range(4)) and len(s.context) == 5 + sum(steps)


def test_rounds_and_fallback(pool):
    script = {"p": [[FOREIGN, KAMIL[0]], [KAMIL[1]], [KAMIL[2]], [FOREIGN]]}
    s = _state()
    SearchRunner(ListGenerator(script), pool, k=1, rounds=2).run([s])
    assert s.lines[0] == KAMIL[0] and s.log[0][:2] == [2, 2]           # second round found it
    assert s.forced == [3] and s.lines[3] == FOREIGN and s.log[3] == [2, 2, False]
    assert s.rounds == 2 + 1 + 1 + 2


def test_fallback_prefers_fewer_new_defects(pool):
    # ʿajuz candidates: wrong meter (bayt fails and rawī) vs right meter, wrong rhyme (rawī only)
    wrong_rhyme = "وَإِذا مَشى في القَومِ كانَ الأَوحَدا"
    s = _state()
    script = {"p": [[KAMIL[0]], [KAMIL[1]], [KAMIL[2]], [FOREIGN, wrong_rhyme]]}
    SearchRunner(ListGenerator(script), pool, k=2, rounds=1).run([s])
    assert s.forced == [3] and s.lines[3] == wrong_rhyme


def test_an_empty_candidate_is_never_preferred_to_a_written_line(pool):
    script = {"p": [["", FOREIGN], [KAMIL[1]], [KAMIL[2]], ["", ""]]}
    s = _state()
    SearchRunner(ListGenerator(script), pool, k=2, rounds=1).run([s])
    assert s.lines[0] == FOREIGN and s.forced == [0, 1, 3]               # the wrong line, not the empty one (its bayt fails)
    assert s.lines[3] == "" and s.empty_lines == 1                       # only empty candidates: counted


def test_exact_length_and_early_end_of_text(pool):
    script = {"p": [[KAMIL[0] + " <eos>"], [KAMIL[1]], [KAMIL[2]], [KAMIL[3]]]}
    s = _state(n=2)
    SearchRunner(ListGenerator(script), pool, k=1, rounds=1).run([s])
    assert len(s.lines) == 2 and s.early_eos == 1


# ---------------------------------------------------------------------------------------------- best-of-N
def test_best_of_n_takes_the_first_valid_sample_and_sums_the_budget():
    v = Verifier(LEX, ignore_diacritics=True, diagnose=False)
    base = {"id": "x", "base_meter": "الكامل", "form": "تام", "requested_num_lines": 4}
    good, bad = "\n".join(KAMIL), "\n".join(KAMIL[:3] + [FOREIGN])
    rows = [dict(base, sample_idx=i, generated_text=t, gen_tokens=10 + i) for i, t in enumerate([bad, good, good, bad])]
    sel = best_of_n(rows, 4, v)
    assert sel["m4"]["selected_sample"] == 1 and sel["m4"]["gen_tokens_total"] == 10 + 11 + 12 + 13
    assert best_of_n(rows, 1, v)["m4"]["selected_sample"] == 0
    assert poem_rank(v, rows[1])[0] is False and poem_rank(v, rows[0])[0] is True
    with pytest.raises(ValueError):
        best_of_n(rows, 5, v)


def test_plan_batches_respects_memory_and_cap():
    pytest.importorskip("torch")
    from m4_decode import plan_batches
    rows = [{"requested_num_lines": n} for n in (2, 40, 4, 2, 40, 4)]
    b = plan_batches(rows, [100] * 6, rows_per_prompt=4, line_tokens=24, budget_tokens=4 * 3 * 1100, cap=3)
    assert sorted(i for x in b for i in x) == list(range(6))
    for x in b:
        width = max(100 + (rows[i]["requested_num_lines"] + 1) * 24 for i in x)
        assert len(x) <= 3 and (len(x) == 1 or len(x) * 4 * width <= 4 * 3 * 1100)
    assert [rows[x[0]]["requested_num_lines"] for x in b] == sorted(rows[x[0]]["requested_num_lines"] for x in b)


# ---------------------------------------------------------------------------------------------- the generator
CHARS = list(" ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءأإآؤئ") + ["َ", "ُ", "ِ", "ْ", "ّ", "ً"]


class CharTokenizer:
    """One token per character; 0 pad, 1 bos, 2 eos, 3 line break."""
    pad_token, eos_token = "<pad>", "</s>"
    pad_token_id, eos_token_id, padding_side = 0, 2, "left"

    def __init__(self):
        self.itos = ["<pad>", "<s>", "</s>", "\n"] + CHARS
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self.stoi[c] for c in text if c in self.stoi])

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.itos[i] for i in ids if not (skip_special_tokens and i < 3))


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    tok = CharTokenizer()
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                                   rope_theta=1_000_000.0, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = transformers.LlamaForCausalLM(cfg).eval()
    return torch, transformers, tok, model


def _args(**kw):
    a = dict(temperature=0.0, top_p=0.9, top_k=0, repetition_penalty=1.08, line_max_tokens=12, seed=1)
    a.update(kw)
    return SimpleNamespace(**a)


class EveryN:
    """Forces a line break after n characters of a line and forbids the end of text (a function of input_ids only)."""

    def __init__(self, torch, n):
        self.torch, self.n = torch, n

    def __call__(self, input_ids, scores):
        for r in range(input_ids.shape[0]):
            row = input_ids[r].tolist()
            since = len(row) - 1 - max((i for i, t in enumerate(row) if t == 3), default=-1)
            scores[r, 2] = -float("inf")
            if since >= self.n:
                scores[r, :] = -float("inf")
                scores[r, 3] = 0.0
        return scores


def _contexts(tok):
    return [tok("ابت ثج ").input_ids, tok("قال الشاعر في وصف الليل ").input_ids, tok("يا ليل").input_ids]


def _plain_greedy(torch, model, tok, contexts, max_new, processors=None, stop_lines=True):
    from m4_decode import LineStop
    from transformers import LogitsProcessorList, StoppingCriteriaList
    width = max(map(len, contexts))
    ids = torch.tensor([[0] * (width - len(c)) + c for c in contexts])
    mask = torch.tensor([[0] * (width - len(c)) + [1] * len(c) for c in contexts])
    stop = torch.zeros(len(tok), dtype=torch.bool)
    stop[[2, 3] if stop_lines else [2]] = True
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, repetition_penalty=1.08,
                             max_new_tokens=max_new, pad_token_id=0, eos_token_id=2,
                             stopping_criteria=StoppingCriteriaList([LineStop(stop)]),
                             logits_processor=LogitsProcessorList(processors or []))
    return out[:, width:].tolist()


def _cut(row):
    return row[: next((i + 1 for i, t in enumerate(row) if t in (2, 3)), len(row))]


@pytest.mark.parametrize("lora", [False, True])
def test_shared_cache_matches_a_full_forward_pass(tiny, lora):
    torch, transformers, tok, model = tiny
    from m4_decode import HFGenerator, inner_model
    if lora:
        peft = pytest.importorskip("peft")
        import copy
        model = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"],
                                                                           init_lora_weights=False))
    gen = HFGenerator(model, tok, torch.device("cpu"), _args())
    ctx = _contexts(tok)
    ids, mask, cache = gen.encode_contexts(ctx)
    pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1]
        step = model(input_ids=ids[:, -1:], attention_mask=mask, position_ids=pos[:, -1:], past_key_values=cache,
                     use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5), (full - step).abs().max()
    assert inner_model(model) is not None


@pytest.mark.parametrize("lora", [False, True])
def test_cached_candidates_equal_plain_generate(tiny, lora):
    torch, transformers, tok, model = tiny
    from m4_decode import HFGenerator
    if lora:
        peft = pytest.importorskip("peft")
        import copy
        model = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"],
                                                                           init_lora_weights=False))
    gen = HFGenerator(model, tok, torch.device("cpu"), _args(line_max_tokens=15))
    ctx = _contexts(tok)
    rows, encoded = gen.sample_lines(ctx, k=2, seed=0)
    assert encoded == [len(c) for c in ctx]
    again, encoded2 = gen.sample_lines(ctx[1:], k=1, seed=0)          # a later round: the encoding is reused
    assert encoded2 == [0, 0] and [_cut(r) for r in again] == [_cut(rows[2]), _cut(rows[4])]
    plain = _plain_greedy(torch, model, tok, ctx, 15)
    for i in range(len(ctx)):
        assert _cut(rows[2 * i]) == _cut(rows[2 * i + 1]) == _cut(plain[i])


def test_stepwise_search_reproduces_one_shot_decoding(tiny):
    """With k = 1 and a checker that accepts everything, the search is plain decoding cut into lines: greedy outputs
    must be identical token for token (contexts, cache reuse, line splitting and the repetition penalty included)."""
    torch, transformers, tok, model = tiny
    from transformers import LogitsProcessorList
    from m4_decode import HFGenerator
    proc = LogitsProcessorList([EveryN(torch, 7)])
    gen = HFGenerator(model, tok, torch.device("cpu"), _args(line_max_tokens=10), logits_processor=proc)

    class AcceptAll:
        def run(self, jobs):
            return [(0, [Check(True, 0, 0.0)]) for _ in jobs]

    prompts = [tok("قال الشاعر\n").input_ids, tok("في الليل والنهار\n").input_ids]
    states = [PoemState(f"p{i}", "kamil", "tam", 4, list(p), len(p)) for i, p in enumerate(prompts)]
    SearchRunner(gen, AcceptAll(), k=1, rounds=1).run(states)
    plain = _plain_greedy(torch, model, tok, prompts, 4 * 8, [EveryN(torch, 7)], stop_lines=False)
    for s, row, p in zip(states, plain, prompts):
        text = tok.decode(row, skip_special_tokens=True)
        assert s.context[s.prompt_len:] == row                       # 4 lines of 7 characters and a line break
        assert s.lines == [x.strip() for x in text.split("\n")[:4]]
        assert s.gen_tokens == 4 * 8 and s.prefill_tokens == sum(len(p) + 8 * j for j in range(4))


def test_candidate_text_ids_and_end_of_text(tiny):
    torch, transformers, tok, model = tiny
    from m4_decode import HFGenerator
    gen = HFGenerator(model, tok, torch.device("cpu"), _args())
    p = tok("قال ").input_ids
    s = PoemState("p", "kamil", "tam", 4, p + tok("ابت ثج\n").input_ids, len(p))
    s.lines = ["ابت ثج"]
    c = gen.candidate(s, tok("دذر زس\nشص").input_ids)
    assert c.text == "دذر زس" and c.ids == tok("دذر زس\n").input_ids and c.tokens == 7 and not c.eos
    c = gen.candidate(s, tok("دذر").input_ids + [2, 0, 0])
    assert c.text == "دذر" and c.eos and c.tokens == 4 and c.ids == tok("دذر\n").input_ids
    c = gen.candidate(s, tok("دذرزسشصضطظ").input_ids)                 # the token limit: the line is cut
    assert c.text == "دذرزسشصضطظ" and c.ids[-1] == 3 and c.tokens == 10


# ---------------------------------------------------------------------------------------------- the summary
def test_mcnemar_exact():
    from m4_report import mcnemar
    assert mcnemar([True] * 5, [True] * 5)["p"] == 1.0
    r = mcnemar([True] * 6 + [False] * 4, [False] * 6 + [False] * 4)      # 6 discordant, all one way
    assert (r["a_only"], r["b_only"]) == (6, 0) and abs(r["p"] - 2 / 64) < 1e-12


def _m4_fixture(tmp_path):
    """A tiny M4 folder: a 4-sample run and a search run on two prompts, scored by the M3 verifier."""
    import m3_rescore as m3
    good, bad = "\n".join(KAMIL), "\n".join(KAMIL[:3] + [FOREIGN])
    base = {"base_meter": "الكامل", "form": "تام", "requested_bayts": 2, "requested_num_lines": 4,
            "reference_completion": good, "parsed_num_lines": 4, "meter": 0.9, "count_adherence": 1.0,
            "meter_target_used": "الكامل", "meter_predicted": "الكامل"}

    def details(text):
        lines = text.split("\n")
        return json.dumps({"per_bayt_details": [{"bayt_index": k, "bayt_text": f"{lines[2 * k]} // {lines[2 * k + 1]}",
                                                 "skipped": False, "pred": "الكامل"} for k in range(2)]}, ensure_ascii=False)
    plan = {"i1_n4": {"a": [bad, good, good, bad], "b": [bad, bad, bad, bad]}, "i2_k2_r1": {"a": [good], "b": [good]}}
    for run, poems in plan.items():
        d = tmp_path / "runs" / run
        d.mkdir(parents=True)
        rows = []
        for pid, texts in poems.items():
            for s_i, t in enumerate(texts):
                m4 = ({"config": {"method": "sample"}, "prefill_tokens": 100, "gen_seconds": 1.0} if run.startswith("i1")
                      else {"config": {"method": "search", "k": 2, "rounds": 1}, "prefill_tokens": 300, "gen_seconds": 2.0,
                            "forced": [] if pid == "a" else [3], "log": [[1, 1, True]] * 3 + [[2, 1, pid == "a"]],
                            "early_eos": 0})
                rows.append(dict(base, id=pid, sample_idx=s_i, generated_text=t, gen_tokens=50 + s_i, m4=m4,
                                 meter_details_json=details(t)))
        for f in ("generations.jsonl", "scored.jsonl"):
            (d / f).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        mode = "sample" if run.startswith("i1") else "search"
        (d / "generations.meta.json").write_text(json.dumps({"mode": mode, "config": plan and (
            {"method": "search", "k": 2, "rounds": 1} if mode == "search" else {"method": "sample"})}))
        m3.score(d / "scored.jsonl", run, tmp_path / "m3", m3.SHIPPED_LEXICON, list(m3.MODES), workers=1, limit=0)
    return tmp_path


def test_summary_selection_budgets_and_checks(tmp_path):
    from m4_report import render, summarize
    res = summarize(_m4_fixture(tmp_path), reps=50)
    c = res["conditions"]
    assert set(n.split(" [")[0] for n in c) == {"I0 single sample", "I1 best-of-2", "I1 best-of-4", "I2 k=2 R=1"}
    i0, b2, b4, i2 = (c[n] for n in sorted(c, key=lambda n: ("I2" in n, c[n]["n"])))
    assert i0["letters"]["strict_valid"][0] == 0.0 and b2["letters"]["strict_valid"][0] == 50.0  # prompt a: sample 1
    assert b4["budget"]["gen_tokens"] == (50 + 51 + 52 + 53) and b2["budget"]["gen_tokens"] == 50 + 51
    assert i2["budget"]["gen_tokens"] == 50 and i2["kind"] == "search"
    assert i2["letters"]["strict_valid"][0] == 100.0
    assert res["checks"][next(n for n in c if "I2" in n)]["unforced_but_invalid"] == 0
    assert all(v in (0, None) for x in res["checks"].values() for v in x.values())
    m = res["matched"][next(n for n in c if "I2" in n)]
    assert m["I1 with at least this budget"]["against"].startswith("I0")         # equal budgets: I0 on both sides
    assert m["I1 with at most this budget"]["against"].startswith("I0")
    assert res["choice"]["k"] == 2 and res["choice"]["rounds"] == 1 and res["choice"]["best_of_n"] == 1
    assert "## Matched compute" in render(res) and "## Integrity checks" in render(res)

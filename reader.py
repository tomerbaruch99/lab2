# Thiis file gets question and the top k clauses and returns the best span

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch, re

RE_REDACT = re.compile(r"(\*{2,}|_{2,}|<omitted>)")

class ExtractiveReader:
    def __init__(self, model_dir="models/extractive_reader"):
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def answer_over_clauses(self, question, clauses, max_len=384, doc_stride=128):
        """
        clauses: list[dict] with keys: text, filename, category, clause_id (אם יש)
        return best dict: {answer_text, score, filename, category, clause_id, start, end}
        """
        best = {"answer_text":"", "score":-1}
        for c in clauses:
            enc = self.tok(question, c["text"], truncation="only_second",
                           max_length=max_len, stride=doc_stride, return_offsets_mapping=True,
                           return_tensors="pt")
            out = self.model(**enc)
            start_scores, end_scores = out.start_logits[0], out.end_logits[0]
            start_idx = int(torch.argmax(start_scores))
            end_idx   = int(torch.argmax(end_scores))
            score = (start_scores[start_idx].item() + end_scores[end_idx].item())/2.0

            offsets = enc["offset_mapping"][0].tolist()
            s_off, e_off = offsets[start_idx][0], offsets[end_idx][1]
            ctx = c["text"]
            ans = ctx[s_off:e_off] if 0 <= s_off < e_off <= len(ctx) else ""
            cand = {
                "answer_text": ans.strip(),
                "score": score,
                "filename": c.get("filename"),
                "category": c.get("category"),
                "clause_id": c.get("clause_id"),
                "start": s_off, "end": e_off,
                "redaction": bool(RE_REDACT.search(ans))
            }
            if score > best["score"] and ans:
                best = cand
        return best if best["score"]>=0 else None

import unittest

from learnloop import scheduler as sch
from learnloop.cards import normalize, quote_in_source
from learnloop.ingest import chunks, html_to_text


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.p = sch.clamp_params({})

    def test_success_grows_stability_and_lapse_shrinks_it(self):
        s1 = sch.update(None, "good", 0, "concept", self.p)
        s2 = sch.update(s1, "good", 3, "concept", self.p)
        s3 = sch.update(s2, "again", 10, "concept", self.p)
        self.assertGreater(s2["stability"], s1["stability"])
        self.assertLess(s3["stability"], s2["stability"])
        self.assertEqual(s3["lapses"], 1)

    def test_interval_hits_target_retention(self):
        st = {"stability": 5.0}
        t = sch.interval_days(st["stability"], 0.9)
        self.assertAlmostEqual(sch.recall_prob(5.0, t), 0.9, places=6)

    def test_clamp_enforces_bounds(self):
        p = sch.clamp_params({"growth": 99, "kind_multiplier": {"number": 0.0}})
        self.assertEqual(p["growth"], sch.BOUNDS["growth"][1])
        self.assertEqual(p["kind_multiplier"]["number"], 0.25)

    def test_replay_scores_only_repeat_reviews_and_respects_score_from(self):
        log = [{"card_id": 1, "ts": 0, "grade": "good"},
               {"card_id": 1, "ts": 3 * 86400, "grade": "good"},
               {"card_id": 1, "ts": 10 * 86400, "grade": "again"}]
        r = sch.replay(log, {1: "concept"}, self.p)
        self.assertEqual(r["n"], 2)
        self.assertEqual(sch.replay(log, {1: "concept"}, self.p, score_from=5 * 86400)["n"], 1)


class QuoteTest(unittest.TestCase):
    SRC = normalize("The mitochondrion is the “powerhouse” of the cell.\nIt makes  ATP through oxidative phosphorylation.")

    def test_verbatim_quote_with_typography_and_whitespace_differences(self):
        self.assertTrue(quote_in_source('the "powerhouse" of the cell. It makes ATP', self.SRC))

    def test_paraphrase_rejected(self):
        self.assertFalse(quote_in_source("mitochondria produce most of the cell's energy", self.SRC))

    def test_too_short_rejected(self):
        self.assertFalse(quote_in_source("the cell", self.SRC))


class IngestTest(unittest.TestCase):
    def test_html_keeps_prose_drops_chrome(self):
        title, text = html_to_text("<html><title>T</title><nav><p>menu</p></nav><p>Hello <b>world</b>.</p>"
                                   "<script>x()</script><table><tr><td><p>cell</p></td></tr></table><li>item one</li></html>")
        self.assertEqual(title, "T")
        self.assertEqual(text, "Hello world.\n\nitem one")

    def test_chunks_split_on_paragraphs(self):
        text = "\n\n".join(["a" * 600] * 10)
        parts = chunks(text, size=2000)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 2000 + 602 for p in parts))
        self.assertEqual("".join(parts).count("a"), 6000)


if __name__ == "__main__":
    unittest.main()

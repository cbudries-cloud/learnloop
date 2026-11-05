"""learnloop command line."""
import argparse
import sys

from . import engine, ingest


def cmd_add(args):
    from .cards import write_cards
    engine.init_db()
    for uri in ingest.expand(args.sources):
        print(f"{uri}")
        try:
            title, text = ingest.load(uri)
        except Exception as e:
            print(f"  skipped: {e}", file=sys.stderr)
            continue
        if len(text.strip()) < 200:
            print("  skipped: not enough text", file=sys.stderr)
            continue
        pieces = ingest.chunks(text)
        cards = []
        for i, piece in enumerate(pieces, 1):
            n = max(3, min(args.per_chunk, len(piece) // 900))
            kept, _ = write_cards(title, piece, n)
            cards += kept
            if len(pieces) > 1:
                print(f"  part {i}/{len(pieces)}: {len(kept)} cards")
        engine.add_source(args.deck, uri, title, len(text), cards)
        print(f"  {len(cards)} cards added to deck '{args.deck}' ({title})")


def cmd_decks(args):
    engine.init_db()
    rows = engine.decks()
    if not rows:
        print("No decks yet. Try: learnloop add <deck> <file-or-url>")
    for r in rows:
        print(f"{r['deck']:<24} {r['cards']:>4} cards  {r['sources']:>3} sources")


def cmd_serve(args):
    from .server import serve
    serve(args.port)


def cmd_bench(args):
    from .bench import run
    run(args.scenarios, args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="learnloop", description="A learning loop that tunes itself to you.")
    sub = ap.add_subparsers(required=True)

    a = sub.add_parser("add", help="write cards from files, folders or URLs into a deck")
    a.add_argument("deck")
    a.add_argument("sources", nargs="+")
    a.add_argument("--per-chunk", type=int, default=10, help="max cards per ~14k characters of source")
    a.set_defaults(fn=cmd_add)

    d = sub.add_parser("decks", help="list decks")
    d.set_defaults(fn=cmd_decks)

    s = sub.add_parser("serve", help="open the review UI")
    s.add_argument("--port", type=int, default=3200)
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("bench", help="check that scheduler tuning helps on simulated learners")
    b.add_argument("--scenarios", nargs="*", default=None)
    b.add_argument("--out", default=None, help="write results JSON here")
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    args.fn(args)

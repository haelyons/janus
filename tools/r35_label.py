"""Round 35, Inception: the agent labels each blind page (PREDICTIONS.md, round 35).

Claude (tools/claude_call.py, no tools), one call per page: the page image and authors/r35_brief_img.md verbatim.
Resumable: a page whose reply exists is skipped.

usage: python -I tools/r35_label.py   (writes runs/r35/img/agent/<code>.json)
"""
import base64
import json
import sys
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import claude_call  # noqa: E402
from r35_pages import parse_md  # noqa: E402

OUT = J / 'runs/r35/img'
SYSTEM = 'Answer in exactly the form requested.'


def main():
    brief = (J / 'authors/r35_brief_img.md').read_text()
    (OUT / 'agent').mkdir(exist_ok=True)
    for page in sorted((OUT / 'blind').glob('I*.png')):
        p = OUT / 'agent' / f'{page.stem}.json'
        if p.exists():
            continue
        img = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                           'data': base64.b64encode(page.read_bytes()).decode()}}
        r = claude_call.converse(SYSTEM, [[img, {'type': 'text', 'text': brief}]])[0]
        assert not r['is_error'], r['text'][:300]
        parsed = parse_md(f'## {page.stem}\n' + r['text']).get(page.stem, {})
        p.write_text(json.dumps({'code': page.stem, 'raw': r['text'], 'parsed': parsed, 'cost_usd': r['cost_usd']},
                                indent=1))
        print(page.stem, parsed, flush=True)


if __name__ == '__main__':
    main()

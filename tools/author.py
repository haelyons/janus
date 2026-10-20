"""Ask a fresh, tool-less model to write one candidate system prompt from a brief."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_call import converse

J = Path(__file__).resolve().parent.parent
kind = sys.argv[1]  # naive | informed
brief = (J / 'authors/brief_core.md').read_text() + '\n' + (J / f'authors/brief_{kind}.md').read_text()
if kind == 'informed':
    brief += '\n' + Path(sys.argv[2]).read_text()
out = converse('', [[{'type': 'text', 'text': brief}]])
(J / f'authors/{kind}.prompt_sent.md').write_text(brief)
(J / f'authors/{kind}.reply.json').write_text(json.dumps(out, indent=1))
name = {'naive': 'C3_naive', 'informed': 'C4_informed'}[kind]
(J / f'arms/{name}.txt').write_text(out[0]['text'].strip() + '\n')
print(kind, out[0]['cost_usd'], len(out[0]['text'].split()), 'words')

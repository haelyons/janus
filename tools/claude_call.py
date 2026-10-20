"""Run a bare Claude conversation: no tools, no settings, hooks, plugins, MCP or memory.

Each turn is a list of Anthropic content blocks; the process stays open across turns so the
conversation is one session. Runs from a neutral scratch directory so nothing on disk is visible.
"""
import json
import os
import subprocess
import time

SANDBOX = '/tmp/claude-1000/-home-hal-dev-interp/4c226950-4aaa-401f-b7d7-179cae20dfa3/scratchpad/sbx'
MODEL = 'claude-opus-5-5'


def converse(system_prompt, turns, model=MODEL, timeout=900):
    os.makedirs(SANDBOX, exist_ok=True)
    cmd = ['claude', '-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
           '--model', model, '--tools', '', '--setting-sources', '', '--strict-mcp-config',
           '--disable-slash-commands', '--no-session-persistence', '--system-prompt', system_prompt]
    p = subprocess.Popen(cmd, cwd=SANDBOX, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    out, t0 = [], time.time()
    try:
        for content in turns:
            if callable(content):  # a turn built from the previous reply; None ends the conversation
                content = content(out)
                if content is None:
                    break
            p.stdin.write(json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}}) + '\n')
            p.stdin.flush()
            res = None
            while res is None:
                line = p.stdout.readline()
                if not line:
                    raise RuntimeError('claude exited: ' + p.stderr.read()[:500])
                ev = json.loads(line)
                if ev.get('type') == 'system' and ev.get('subtype') == 'init' and ev.get('tools'):
                    raise RuntimeError(f'tools present: {ev["tools"]}')
                if ev.get('type') == 'result':
                    res = ev
                if time.time() - t0 > timeout:
                    raise TimeoutError('turn timed out')
            out.append({'text': res.get('result', ''), 'cost_usd': res.get('total_cost_usd'),
                        'usage': res.get('usage'), 'is_error': res.get('is_error'), 'ms': res.get('duration_ms')})
    finally:
        try:
            p.stdin.close()
            p.wait(timeout=60)
        except Exception:
            p.kill()
    return out


def parse_json(text):
    """the last JSON object in a reply that carries an expected key (bare, fenced, or followed by prose)"""
    dec, found = json.JSONDecoder(), []
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                found.append(obj)
    for obj in reversed(found):
        if 'class' in obj or 'predictions' in obj or 'edits' in obj:
            return obj
    if found:
        return found[-1]
    raise ValueError('no JSON object in reply')

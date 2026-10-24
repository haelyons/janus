"""Round 31A2 runner: tools/vlm_answer.py unchanged, with vllm.LLM given tensor_parallel_size by patching it at run
time (no other argument, prompt or sampling setting changes), for a model larger than one GPU.

usage: python -I tools/r31_tp.py --tp 4 <tools/vlm_answer.py arguments>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    i = sys.argv.index('--tp')
    tp = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    import vllm
    orig = vllm.LLM

    def LLM(*a, **k):
        assert 'tensor_parallel_size' not in k, k
        return orig(*a, tensor_parallel_size=tp, **k)
    vllm.LLM = LLM
    import vlm_answer
    sys.argv[0] = 'vlm_answer.py'
    vlm_answer.main()


if __name__ == '__main__':
    main()

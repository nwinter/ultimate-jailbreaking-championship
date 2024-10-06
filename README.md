# Jailbreaking Experimentation with Inspect

This code was written to play around with various jailbreaking techniques ahead of the Gray Swan AI Ultimate Jailbreaking Competition. I've written that up [here on my blog](https://www.nickwinter.net/posts/my-experiences-in-gray-swan-ais-ultimate-jailbreaking-championship).

This is disorganized experimental mix-and-match code that helped me do a few things:
1. Test 10 off-the-shelf jailbreaking prompt templates on 15 sample HarmBench target behaviors against 18 LLMs from a range of providers, automatically scoring their outputs with 4 rating methods
2. Quickly pipe several combinations of Behavior x Attack x Paraphrase into a Google Sheet
3. Play around with running open-weight GCG attacks
4. Convert competition chat log files into spreadsheet format

It's most useful as a quick demo of how one might use the UK AISI's [Inspect](https://inspect.ai-safety-institute.org.uk/) framework, which is a fantastic tool for doing all sorts of evals, whether on AI capabilities elicitation or red-teaming techniques. You may just want to read their docs, which will have more canonical ways of using their tooling, rather than starting from this baseline.

Don't imagine you will want to run this, but if you do, use `uv` to manage your Python environment, then `uv run jailbreak.py`, `uv run gcg.py`, etc. You'll also want to put API keys for some subset of OpenAI, Anthropic, Together.ai, HuggingFace, and Google into .env (following .env.example format).

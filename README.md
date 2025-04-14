# Overview

Given a set of files, this tool finds the top _k_ according to any given criteria, using an LLM as the judge.

It's intended for **small datasets** and **small _k_**; e.g., finding the top 10 out of a few hundred files. Comparisons are done **pairwise**: the LLM is given two documents at a time and asked to pick the better one according to the specified criteria. A single-elimination tournament is used to determine the overall "best" file, with additional rounds to determine the runners-up. For a dataset of _n_ files, the tool has to invoke the LLM approximately `(n-1) + (k-1)*log_2(n)` times.

Currently, only text files are supported (no images/PDFs/etc).

The supported model APIs are Ollama and Anthropic (Claude).

# Usage

First, put all the files you want to rank into one folder. Basic usage of the tool looks like this:

```
rank-files 'Each document is a book review. The best document is the book review that contains the most thoughtful original content, as opposed to just summarizing or quoting the book.' path/to/input-folder -k 10
```

However, you'll probably need extra setup based on which model and model provider you want to use.

## Ollama

By default the tool will assume you have Ollama locally. You can use a remote Ollama instance by setting the `OLLAMA_HOST` environment variable to the appropriate URL.

You must have whatever model you want to use installed in Ollama ahead of time. By default the tool tries to use `gemma3:4b`, which you can install via `ollama pull gemma3:4b`. However, this model may not be powerful enough for use cases like the one in the example above. You can set the `RANK_FILES_MODEL` environment variable to use a different model, e.g.:

```
RANK_FILES_MODEL=llama3.3:70b rank-files 'Each document is a book review. The best document is the book review that contains the most thoughtful original content, as opposed to just summarizing or quoting the book.' path/to/input-folder -k 10
```

## Claude

Alternatively, you can use **Claude** by setting `ANTHROPIC_API_KEY`, `RANK_FILES_PROVIDER`, and `RANK_FILES_MODEL`. Remember, this costs money and the number of API invocations grows superlinearly; make sure you know what you're doing.

```
ANTHROPIC_API_KEY=... RANK_FILES_PROVIDER=anthropic RANK_FILES_MODEL='claude-3-5-haiku-latest' rank-files 'Each document is a book review. The best document is the book review that contains the most thoughtful original content, as opposed to just summarizing or quoting the book.' path/to/input-folder -k 10
```

## Caching

You will notice a file named `rank-files-cache.sqlite3` created in the current directory when you run the tool. This stores hashes of prompts and the responses received for them, so that the tool won't ask the same model to compare the same two files twice.

This means that if the tool is interrupted, no important work is lost—you can rerun it again with the same parameters (the criteria must be exactly the same) and it will use the cached results for any comparisons that were already performed.

If you want the cache to go somewhere else, set the `RANK_FILES_CACHE` environment variable to the desired path and filename; or set it to `:memory:` if you don't want it at all.

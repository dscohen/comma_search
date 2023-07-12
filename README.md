# comma_search
# =============

# Download indexed/raw documents from the July 9th-ish crawl of the Comma Discord and untar/unzip them into a directory called `~/data/`(if you want to run without changing file paths).

Since it's pre-indexed, you don't have to do any initial encoding. Just run `python3 search.py` and you'll be prompted for a query. Enter a query and it'll return the top 10 results. The first 10 results are before any post-processing, and the second 10 results are after post-processing. The post-processing is just a simple centroid based PRF.

This is meant to be a fast and dirty search to get you the exact match text to paste into discord search to get the context. 

Requirements:
- Python 3.6+
- Pyserini (look at repo for installation instructions: https://github.com/castorini/pyserini/blob/master/docs/installation.md)
- transformers
- sklearn
- pytorch
- faiss-cpu

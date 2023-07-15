# comma_search


Download indexed/raw documents from the July 9th-ish crawl of the Comma Discord and untar/unzip them into a directory called `~/data/`(if you want to run without changing file paths): https://drive.google.com/file/d/16Sh-mxHowLq_oCLR9zVoq4WF0sMXD4Yz/view?usp=sharing

Since it's pre-indexed, you don't have to do any initial encoding. Just run `python3 search.py` and you'll be prompted for a query. Enter a query and it'll return the top 10 results. The first 10 results are before any post-processing, and the second 10 results are after post-processing. The post-processing is just a simple centroid based PRF.

This is meant to be a fast and dirty search to get you the exact match text to paste into discord search to get the context. 

It will take a few minutes the first time you run it to build the inverse document frequency needed for the PRF. After that, it will only take a couple seconds to start. You can just to print more or less by editing the print statement in main().

Requirements:
- Python 3.6+
- Pyserini (look at repo for installation instructions: https://github.com/castorini/pyserini/blob/master/docs/installation.md)
- JDK, if you're using conda: conda install -c conda-forge openjdk=11
- transformers
- sklearn
- pytorch
- faiss-cpu

## Fixed-length Sequence Memory Evolution ##

`evolve.py` automatically selects images to breed based on how different they are from any images previously
seen during the current run.  The 'novelty' of an image is the minimum Euclidean distance from that image to 
each image in the set of archived images.  The most novel image in each generation is always added to the archive,
and other images are randomly added with low probability regardless of their novelty. 
 
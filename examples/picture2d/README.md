## 2D Image Evolution ##

These examples demonstrate one approach to evolve 2D patterns using NEAT.  These sorts of networks are 
referred to as "Compositional Pattern-Producing Networks" in the NEAT literature.

## Interactive Example ##

`evolve.py` is an example that amounts to an offline picbreeder.org without any nice features. :)

Left-click on thumbnails to pick images to breed for next generation, right-click to
render a high-resolution version of an image.  Genomes and images chosen for breeding
and rendering are saved to disk.

## Non-Interactive

`novelty.py` automatically selects images to breed based on how different they are from any images previously
seen during the current run.  The 'novelty' of an image is the minimum Euclidean distance from that image to 
each image in the set of archived images.  The most novel image in each generation is always added to the archive,
and other images are randomly added with low probability regardless of their novelty. 
 
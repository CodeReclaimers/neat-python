Frequently Asked Questions
==========================

Algorithm Choice
----------------

When should I use NEAT vs. backpropagation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use NEAT when:**

- You don't know the optimal network architecture beforehand
- You want minimal/compact solutions (NEAT starts simple and adds complexity)
- Your problem has sparse or delayed rewards (difficult to compute gradients)
- You need interpretable networks that can be visualized
- You're evolving behavior for agents/robots (evolutionary robotics)

**Use backpropagation (gradient descent) when:**

- You know the appropriate architecture (e.g., CNNs for images)
- You have large labeled datasets
- You need very deep networks (100+ layers)
- Training speed is critical and you have GPU resources
- The problem is well-suited to standard architectures

**Example comparison:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Aspect
     - NEAT
     - Backpropagation
   * - Architecture
     - Evolved automatically
     - Hand-designed
   * - Training data
     - Minimal (fitness only)
     - Large labeled datasets
   * - Network size
     - 10-100 nodes typical
     - 1000s-millions of weights
   * - Interpretability
     - High (small networks)
     - Low (black boxes)
   * - Computation
     - CPU-based
     - GPU-accelerated

**See also:** :doc:`neat_overview` for how NEAT works.

Can NEAT do deep learning?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Short answer:** Not really, and you shouldn't try.

NEAT excels at evolving **small to medium networks** (10-100 nodes). Its strength is finding minimal, elegant solutions to problems. Deep learning typically involves networks with thousands to millions of parameters.

**Why NEAT isn't for deep learning:**

1. **Scalability:** Evolution doesn't scale well to millions of parameters
2. **Speed:** Gradient descent is orders of magnitude faster for large networks
3. **Architecture:** Deep learning relies on specific architectures (CNNs, Transformers) that work well with gradients

**NEAT's sweet spot:** Control problems, game playing, robotics, function approximation with 2-20 inputs and 1-10 outputs.

**See also:** :doc:`config_essentials` for typical problem sizes.

Should I use feedforward or recurrent networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use feedforward when:**

- Each input → output mapping is independent
- No temporal dependencies or memory needed
- Problem is stateless
- **Examples:** XOR, classification, function approximation

**Use recurrent when:**

- Current output depends on past inputs
- Problem involves sequences or time-series
- Agent needs memory of previous states
- **Examples:** Game playing, control, sequence prediction, navigation

**Configuration:**

.. code-block:: ini

   [DefaultGenome]
   # Feedforward
   feed_forward = True
   
   # Recurrent (allows cycles/memory)
   feed_forward = False

**Tip:** Start with feedforward. Only use recurrent if your problem clearly needs memory.

**See also:** :doc:`ctrnn` for continuous-time recurrent networks.

Performance & Scaling
---------------------

How many generations is typical?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Highly problem-dependent:**

- **Simple problems** (XOR, simple control): 50-150 generations
- **Medium problems** (pole balancing, basic games): 100-500 generations
- **Complex problems** (multi-objective, complex behaviors): 500-5000+ generations

**Rule of thumb:** Start with 300 generations and adjust.

**What affects convergence speed:**

- Population size (larger = faster convergence, slower per generation)
- Problem difficulty
- Fitness function design (clear gradients help)
- Mutation rates
- Initial configuration

**Monitoring progress:**

.. code-block:: python

   stats = neat.StatisticsReporter()
   p.add_reporter(stats)
   
   # Check if fitness is improving
   # If stuck after 50-100 generations, adjust parameters

**See also:** :doc:`troubleshooting` if evolution is stuck.

What's a good population size?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default recommendation:** Start with 150 (from the original NEAT paper).

**Adjust based on problem complexity:**

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Problem Type
     - Population Size
     - Trade-off
   * - Very simple (XOR)
     - 50-100
     - Fast iterations
   * - Simple-Medium
     - 150-200
     - Balanced
   * - Medium-Complex
     - 200-500
     - More exploration
   * - Very complex
     - 500-1000+
     - Thorough search

**Population size effects:**

Larger populations:
- ✅ More diversity
- ✅ Better exploration
- ✅ More robust solutions
- ❌ Slower per generation
- ❌ More memory usage

Smaller populations:
- ✅ Faster iterations
- ✅ Less memory
- ❌ Risk of premature convergence
- ❌ Less diversity

**See also:** :doc:`config_essentials` for population size guidance.

Can I use GPUs?
~~~~~~~~~~~~~~~

**NEAT-Python is CPU-only.** The evolutionary algorithm itself doesn't benefit from GPU acceleration.

**However:**

- You **can** use GPUs in your fitness function (e.g., if running neural network simulations with PyTorch)
- The evolved networks themselves are small and fast on CPU
- Use ``ParallelEvaluator`` to utilize multiple CPU cores

**Example with GPU fitness evaluation:**

.. code-block:: python

   import torch
   import neat
   
   # Your fitness function can use GPU
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Use GPU for expensive computation
           with torch.cuda.device(0):
               fitness = gpu_based_evaluation(net)  # Your GPU code
           
           genome.fitness = fitness

**Why NEAT doesn't need GPU:**

- Networks are small (10-100 nodes)
- Forward pass is simple (no backpropagation)
- Evolution is inherently parallel (use multiple CPU cores)

**See also:** :doc:`cookbook` section on parallel evaluation.

How do I speed up evolution?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Use ParallelEvaluator** (easiest speedup):

.. code-block:: python

   import multiprocessing
   with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as pe:
       winner = p.run(pe.evaluate, 300)

**2. Optimize your fitness function:**

- Cache expensive computations
- Vectorize operations (NumPy)
- Profile and remove bottlenecks

**3. Reduce population size:**

50% population = ~2x speedup, but less thorough exploration.

**4. Simplify network evaluation:**

- Use feedforward when possible (faster than recurrent)
- Reduce number of fitness test cases

**5. Early termination:**

.. code-block:: python

   # Stop when good enough, don't wait for perfect
   [NEAT]
   fitness_threshold = 3.9  # Not 4.0

**6. Tune configuration:**

- Higher mutation rates = faster exploration (but potentially premature convergence)
- Fewer species = faster convergence

**Typical speedups:**
- ParallelEvaluator (4 cores): 3-4x faster
- Optimized fitness function: 2-10x faster
- Combined: 6-40x faster

Configuration
-------------

Why do my networks keep getting bigger?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reason:** NEAT has a "complexification" bias—it starts simple and adds structure over time. Without constraints, networks can grow large.

**Solutions:**

**1. Lower mutation rates:**

.. code-block:: ini

   [DefaultGenome]
   conn_add_prob = 0.3      # Default: 0.5
   node_add_prob = 0.1      # Default: 0.2

**2. Add complexity penalty:**

.. code-block:: python

   genome.fitness = task_fitness - 0.01 * len(genome.connections)

**3. Limit structural mutations:**

.. code-block:: ini

   [DefaultGenome]
   single_structural_mutation = true

**See also:** :doc:`cookbook` section on "Control Network Complexity".

What activation function should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default: sigmoid** (outputs in [0, 1])

**Common choices:**

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Function
     - Output Range
     - Best For
   * - sigmoid
     - [0, 1]
     - Probabilities, classification, default choice
   * - tanh
     - [-1, 1]
     - Control problems, normalized outputs
   * - relu
     - [0, ∞)
     - Function approximation, fast computation
   * - identity
     - (-∞, ∞)
     - Linear problems, custom scaling

**Configuration:**

.. code-block:: ini

   [DefaultGenome]
   # Fixed activation
   activation_default = tanh
   activation_mutate_rate = 0.0
   activation_options = tanh
   
   # Or allow evolution to choose
   activation_default = sigmoid
   activation_mutate_rate = 0.1
   activation_options = sigmoid tanh relu

**Tip:** Match activation to your problem's output range. If you need [-1, 1] outputs, use tanh!

**See also:** :doc:`activation` for all available functions.

How do I set fitness_threshold?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: Problem-dependent (best)**

Based on what "perfect" performance means:

.. code-block:: ini

   # XOR: 4 test cases, each can contribute 1.0 error
   # Perfect = 4.0, "good enough" = 3.9
   fitness_threshold = 3.9
   
   # Game score: maybe 500 is winning
   fitness_threshold = 500

**Method 2: Run without threshold first**

.. code-block:: ini

   no_fitness_termination = True  # Ignores fitness_threshold

Run for fixed generations, see what best fitness achieved, then set threshold slightly below that.

**Method 3: Unreachable threshold**

.. code-block:: ini

   fitness_threshold = 1000000  # Unrealistic
   # Just run for N generations

**Tip:** Set threshold to 95-98% of "perfect" to avoid waiting for marginal improvements.

Understanding Behavior
-----------------------

What does speciation do?
~~~~~~~~~~~~~~~~~~~~~~~~~

**Speciation protects innovation** by grouping similar genomes together.

**Key concept:** A structural mutation (adding a node/connection) often decreases fitness initially. Without protection, this innovation would be eliminated before it can be refined.

**How it works:**

1. NEAT groups genomes by genetic similarity
2. Fitness competition is strongest **within** species, not between species
3. New innovations get time to improve before competing globally

**Configuration:**

.. code-block:: ini

   [DefaultSpeciesSet]
   compatibility_threshold = 3.0  # Lower = more species

**Benefits:**

- Protects topological innovations
- Maintains diversity
- Prevents premature convergence
- Allows parallel exploration of different solutions

**See also:** :doc:`neat_overview` for algorithm details.

Why does fitness sometimes decrease?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Normal reasons:**

**1. Best genome not selected for reproduction**

Elitism preserves some best genomes, but not all:

.. code-block:: ini

   [DefaultReproduction]
   elitism = 2  # Only top 2 survive unchanged

**2. Stochastic fitness evaluation**

If fitness has randomness, same genome can get different scores.

**Solution:** Average over multiple trials:

.. code-block:: python

   fitness_values = [evaluate(net) for _ in range(3)]
   genome.fitness = sum(fitness_values) / len(fitness_values)

**3. Species removed due to stagnation**

Top species can be removed if it stagnates, taking its best genome with it.

**Problematic reasons:**

**4. Bug in fitness function**

- Using random values incorrectly
- Not setting fitness consistently

**5. Fitness not deterministic**

Make sure same network always gets same fitness for same inputs.

**See also:** :doc:`troubleshooting` for debugging evolution.

What's the difference between nodes and connections?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Nodes** = **Neurons** in the network

- Input nodes: receive external input (keys: -1, -2, ...)
- Output nodes: produce network output (keys: 0, 1, ...)
- Hidden nodes: internal processing (keys: 1+, assigned during evolution)

**Connections** = **Synapses** between neurons

- Have a weight (strength of connection)
- Have innovation number (for tracking)
- Can be enabled/disabled

**Example:**

.. code-block:: text

   XOR network might have:
   - 2 input nodes (-1, -2)
   - 1 output node (0)
   - 0-2 hidden nodes (evolved)
   - 2-6 connections (evolved)

**In code:**

.. code-block:: python

   print(f"Nodes: {len(genome.nodes)}")
   print(f"Connections: {len(genome.connections)}")
   
   # Network size reported as (nodes, connections)
   # e.g., "size: (3, 5)" = 3 nodes, 5 connections

Integration
-----------

How do I use NEAT with OpenAI Gym?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Basic pattern:**

.. code-block:: python

   import gym
   import neat
   
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           env = gym.make('CartPole-v1')
           
           fitness = 0.0
           observation = env.reset()
           
           for _ in range(500):  # Max steps
               # Get action from network
               action = net.activate(observation)
               
               # Convert to gym action format
               action = int(action[0] > 0.5)  # For discrete actions
               
               # Step environment
               observation, reward, done, info = env.step(action)
               fitness += reward
               
               if done:
                   break
           
           env.close()
           genome.fitness = fitness

**Config considerations:**

.. code-block:: ini

   [DefaultGenome]
   num_inputs = 4   # CartPole has 4 observations
   num_outputs = 1  # Single action (or 2 for softmax)
   activation_default = tanh  # Good for control

**See example:** ``examples/openai-lander/`` in the repository.

Can I use NEAT with PyTorch/TensorFlow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** You can use them in your fitness function, but the evolved networks themselves are NEAT networks, not PyTorch/TF models.

**Common pattern: GPU-accelerated fitness evaluation:**

.. code-block:: python

   import torch
   import neat
   
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           # NEAT network (on CPU)
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Use PyTorch for expensive computations
           inputs_tensor = torch.tensor(inputs, device='cuda')
           results = expensive_gpu_computation(inputs_tensor)
           
           # Evaluate NEAT network on results
           genome.fitness = evaluate(net, results.cpu().numpy())

**Can I export to PyTorch/TensorFlow?**

Not directly. NEAT networks have dynamic topology that doesn't map cleanly to fixed architectures. However, you can:

1. Manually implement the evolved topology in PyTorch/TF
2. Use :doc:`network_export` to export to JSON, then build equivalent model
3. Treat NEAT network as a black-box controller

**See also:** :doc:`network_export` for export options.

How do I export evolved networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. JSON export (recommended):**

.. code-block:: python

   import neat.export
   
   # Export network structure and weights
   neat.export.save_json(config, winner, 'winner-network.json')

**2. Graphviz visualization:**

.. code-block:: python

   import visualize  # From examples/xor/
   
   visualize.draw_net(config, winner, view=True,
                      node_names={-1: 'Input1', -2: 'Input2', 0: 'Output'})

**3. Manual extraction:**

.. code-block:: python

   # Get network structure
   net = neat.nn.FeedForwardNetwork.create(winner, config)
   
   # Access nodes and connections
   for key, node in winner.nodes.items():
       print(f"Node {key}: bias={node.bias}, activation={node.activation}")
   
   for key, conn in winner.connections.items():
       print(f"Connection {key}: weight={conn.weight}, enabled={conn.enabled}")

**See also:** :doc:`network_export` for detailed export options.

Academic
--------

How do I cite NEAT-Python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cite both the library and the original NEAT paper:**

**NEAT-Python library:**

.. code-block:: bibtex

   @misc{neat-python,
     author = {CodeReclaimers, LLC},
     title = {{NEAT-Python}},
     howpublished = {\\url{https://github.com/CodeReclaimers/neat-python}},
     year = {2015--2025}
   }

**Original NEAT paper:**

.. code-block:: bibtex

   @article{stanley2002evolving,
     title={Evolving neural networks through augmenting topologies},
     author={Stanley, Kenneth O and Miikkulainen, Risto},
     journal={Evolutionary computation},
     volume={10},
     number={2},
     pages={99--127},
     year={2002},
     publisher={MIT Press}
   }

**In text:** "We used NEAT-Python (CodeReclaimers), an implementation of the NEAT algorithm (Stanley & Miikkulainen, 2002)."

What's the difference from the original NEAT?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NEAT-Python **closely follows** the original NEAT paper (Stanley & Miikkulainen, 2002), with a few differences:

**Implemented from the paper:**

- ✅ Innovation number tracking (v1.0.0+)
- ✅ Speciation with fitness sharing
- ✅ Structural mutation (add node/connection)
- ✅ Crossover of matching genes
- ✅ Complexification (start simple, add structure)

**Differences/Extensions:**

1. **Pure Python implementation** (original was C++)
2. **Multiple activation functions** (original used sigmoid only)
3. **Configurable reproduction** (original had fixed parameters)
4. **Additional network types:**
   - Recurrent networks
   - CTRNN (continuous-time)
   - IZNN (Izhikevich spiking)
5. **Enhanced features:**
   - Checkpointing
   - Statistics reporting
   - Network export
   - Parallel evaluation

**Key improvement in v1.0.0:**

Innovation number tracking was rewritten to fully match the paper's specification, improving crossover accuracy.

**See also:** 
- :doc:`innovation_numbers` for v1.0.0 improvements
- :doc:`neat_overview` for algorithm description

More Questions?
---------------

**Check these resources:**

- :doc:`cookbook` - Practical how-to recipes
- :doc:`troubleshooting` - Diagnostic help
- :doc:`config_essentials` - Configuration guide
- :doc:`xor_example` - Complete working example

**Still need help?**

- GitHub Issues: https://github.com/CodeReclaimers/neat-python/issues
- Original NEAT website: http://www.cs.ucf.edu/~kstanley/

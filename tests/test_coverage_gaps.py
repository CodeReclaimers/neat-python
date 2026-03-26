"""
Tests targeting specific coverage gaps across the neat-python codebase.

Each test class addresses uncovered code paths in a specific module,
focusing on real features rather than just increasing the metric.
"""

import math
import os
import pickle
import random
import warnings

import pytest

import neat
from neat.activations import (
    elu_activation, lelu_activation, selu_activation,
)
from neat.aggregations import AggregationFunctionSet, validate_aggregation, InvalidAggregationFunction
from neat.config import ConfigParameter, UnknownConfigItemError
from neat.export.json_format import (
    validate_json, is_builtin_activation, is_builtin_aggregation,
    get_function_info,
)
from neat.export.exporters import (
    _get_neat_version, export_feedforward, export_recurrent,
    export_ctrnn, export_iznn,
)
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.innovation import InnovationTracker
from neat.iznn import IZNeuron, IZNN
from neat.reproduction import DefaultReproduction


LOCAL_DIR = os.path.dirname(__file__)


def _load_config(name='test_configuration'):
    path = os.path.join(LOCAL_DIR, name)
    return neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, path,
    )


# ──────────────────────────────────────────────────────────────────────
# Activation functions: ELU, Leaky ReLU, SELU (lines 36, 40-41, 45-47)
# ──────────────────────────────────────────────────────────────────────

class TestActivationFunctions:
    """Cover the three activation functions that were never called."""

    def test_elu_positive(self):
        assert elu_activation(2.0) == 2.0

    def test_elu_negative(self):
        result = elu_activation(-1.0)
        assert result == pytest.approx(math.exp(-1.0) - 1)

    def test_elu_zero(self):
        assert elu_activation(0.0) == 0.0

    def test_lelu_positive(self):
        assert lelu_activation(3.0) == 3.0

    def test_lelu_negative(self):
        result = lelu_activation(-2.0)
        assert result == pytest.approx(0.005 * -2.0)

    def test_lelu_zero(self):
        assert lelu_activation(0.0) == 0.0

    def test_selu_positive(self):
        lam = 1.0507009873554804934193349852946
        assert selu_activation(1.5) == pytest.approx(lam * 1.5)

    def test_selu_negative(self):
        lam = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        result = selu_activation(-1.0)
        expected = lam * alpha * (math.exp(-1.0) - 1)
        assert result == pytest.approx(expected)

    def test_selu_zero(self):
        assert selu_activation(0.0) == 0.0


# ──────────────────────────────────────────────────────────────────────
# Aggregation: deprecation warning on __getitem__ (lines 98-100)
# ──────────────────────────────────────────────────────────────────────

class TestAggregationDeprecation:
    def test_getitem_warns(self):
        afs = AggregationFunctionSet()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func = afs['sum']
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'get' in str(w[0].message)
        # Should still return the right function
        assert func is afs.get('sum')

    def test_validate_aggregation_bad_signature(self):
        """Aggregation with wrong signature should be rejected (line 63+)."""
        def bad_agg(a, b):
            return a + b
        with pytest.raises(InvalidAggregationFunction):
            validate_aggregation(bad_agg)


# ──────────────────────────────────────────────────────────────────────
# ConfigParameter.interpret() edge cases (lines 52-53, 57, 72-75, 90, 95-99)
# ──────────────────────────────────────────────────────────────────────

class TestConfigParameterInterpret:
    def test_missing_required_with_section_name(self):
        """Error message includes section name (line 72-73)."""
        p = ConfigParameter('some_param', int)
        with pytest.raises(RuntimeError, match=r"\[MySection\].*some_param"):
            p.interpret({}, section_name='MySection')

    def test_missing_required_without_section_name(self):
        """Error message without section (line 74-75)."""
        p = ConfigParameter('some_param', int)
        with pytest.raises(RuntimeError, match="some_param"):
            p.interpret({})

    def test_bool_must_be_true_or_false(self):
        """Non-boolean string raises (line 90)."""
        p = ConfigParameter('flag', bool)
        with pytest.raises(RuntimeError, match="flag"):
            p.interpret({'flag': 'maybe'})

    def test_unexpected_type(self):
        """Unsupported value_type raises (line 99)."""
        p = ConfigParameter('x', dict)  # dict is not a supported type
        with pytest.raises(RuntimeError, match="Unexpected configuration type"):
            p.interpret({'x': 'hello'})

    def test_parse_unexpected_type(self):
        """parse() with unsupported type raises (line 57)."""
        from configparser import ConfigParser
        cp = ConfigParser()
        cp.read_string("[S]\nx = hello\n")
        p = ConfigParameter('x', dict)
        with pytest.raises(RuntimeError, match="Unexpected configuration type"):
            p.parse('S', cp)


# ──────────────────────────────────────────────────────────────────────
# Config: unknown items detection (lines 139, 193, 206)
# ──────────────────────────────────────────────────────────────────────

class TestConfigUnknownItems:
    def test_unknown_neat_items_detected(self):
        """Typos / extra items in config are caught."""
        # bad_configuration files already exist for this. Let's verify one.
        with pytest.raises((UnknownConfigItemError, RuntimeError)):
            _load_config('bad_configuration0')


# ──────────────────────────────────────────────────────────────────────
# JSON export validation (json_format.py lines 141-175)
# ──────────────────────────────────────────────────────────────────────

class TestJsonValidation:
    def _valid_data(self):
        return {
            'format_version': '1.0',
            'network_type': 'feedforward',
            'metadata': {'created_timestamp': '2025-01-01T00:00:00'},
            'topology': {
                'num_inputs': 2,
                'num_outputs': 1,
                'input_keys': [-1, -2],
                'output_keys': [0],
            },
            'nodes': [
                {'id': 0, 'type': 'output', 'activation': {'name': 'sigmoid'},
                 'aggregation': {'name': 'sum'}, 'bias': 0.0, 'response': 1.0},
                {'id': -1, 'type': 'input', 'activation': {'name': 'identity'},
                 'aggregation': {'name': 'none'}, 'bias': 0.0, 'response': 1.0},
            ],
            'connections': [
                {'from': -1, 'to': 0, 'weight': 0.5, 'enabled': True},
            ],
        }

    def test_valid_data_passes(self):
        assert validate_json(self._valid_data()) is True

    def test_missing_top_level_field(self):
        data = self._valid_data()
        del data['network_type']
        with pytest.raises(ValueError, match="Missing required field"):
            validate_json(data)

    def test_invalid_network_type(self):
        data = self._valid_data()
        data['network_type'] = 'transformer'
        with pytest.raises(ValueError, match="Invalid network_type"):
            validate_json(data)

    def test_missing_topology_field(self):
        data = self._valid_data()
        del data['topology']['num_inputs']
        with pytest.raises(ValueError, match="Missing required topology field"):
            validate_json(data)

    def test_nodes_not_list(self):
        data = self._valid_data()
        data['nodes'] = "not a list"
        with pytest.raises(ValueError, match="nodes must be a list"):
            validate_json(data)

    def test_node_missing_id(self):
        data = self._valid_data()
        data['nodes'] = [{'type': 'input'}]
        with pytest.raises(ValueError, match="missing 'id'"):
            validate_json(data)

    def test_node_missing_type(self):
        data = self._valid_data()
        data['nodes'] = [{'id': 0}]
        with pytest.raises(ValueError, match="missing 'type'"):
            validate_json(data)

    def test_node_invalid_type(self):
        data = self._valid_data()
        data['nodes'] = [{'id': 0, 'type': 'encoder'}]
        with pytest.raises(ValueError, match="invalid type"):
            validate_json(data)

    def test_connections_not_list(self):
        data = self._valid_data()
        data['connections'] = 42
        with pytest.raises(ValueError, match="connections must be a list"):
            validate_json(data)

    def test_connection_missing_field(self):
        data = self._valid_data()
        data['connections'] = [{'from': -1, 'to': 0, 'weight': 0.5}]  # no 'enabled'
        with pytest.raises(ValueError, match="missing 'enabled'"):
            validate_json(data)


# ──────────────────────────────────────────────────────────────────────
# get_function_info with None and custom functions (lines 75, 88, 104-105)
# ──────────────────────────────────────────────────────────────────────

class TestGetFunctionInfo:
    def test_none_activation(self):
        info = get_function_info(None, 'activation')
        assert info == {"name": "none", "custom": False}

    def test_none_aggregation(self):
        info = get_function_info(None, 'aggregation')
        assert info == {"name": "none", "custom": False}

    def test_builtin_activation(self):
        info = get_function_info(neat.activations.sigmoid_activation, 'activation')
        assert info['name'] == 'sigmoid'
        assert info['custom'] is False

    def test_custom_function_detected(self):
        def my_activation(z):
            return z
        info = get_function_info(my_activation, 'activation')
        assert info['custom'] is True

    def test_is_builtin_activation_no_module(self):
        """Function with no detectable module returns False (line 74-75)."""
        # Built-in functions like 'abs' have module builtins, not neat.activations
        assert is_builtin_activation(abs) is False

    def test_is_builtin_aggregation_no_module(self):
        assert is_builtin_aggregation(abs) is False


# ──────────────────────────────────────────────────────────────────────
# Export metadata propagation (exporters.py lines 162, 236, 274, 316)
# ──────────────────────────────────────────────────────────────────────

class TestExportMetadata:
    """Verify metadata dict is merged into exported data for all network types."""

    def _make_feedforward_net(self):
        config = _load_config()
        p = neat.Population(config)
        genome = list(p.population.values())[0]
        genome.fitness = 1.0
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def _make_recurrent_net(self):
        config = _load_config()
        p = neat.Population(config)
        genome = list(p.population.values())[0]
        genome.fitness = 1.0
        return neat.nn.RecurrentNetwork.create(genome, config)

    def test_feedforward_metadata(self):
        net = self._make_feedforward_net()
        data = export_feedforward(net, metadata={'fitness': 42.0, 'generation': 5})
        assert data['metadata']['fitness'] == 42.0
        assert data['metadata']['generation'] == 5
        assert 'created_timestamp' in data['metadata']

    def test_recurrent_metadata(self):
        net = self._make_recurrent_net()
        data = export_recurrent(net, metadata={'genome_id': 99})
        assert data['metadata']['genome_id'] == 99

    def test_ctrnn_metadata(self):
        config = _load_config()
        p = neat.Population(config)
        genome = list(p.population.values())[0]
        genome.fitness = 1.0
        net = neat.ctrnn.CTRNN.create(genome, config)
        data = export_ctrnn(net, metadata={'fitness': 10.0})
        assert data['metadata']['fitness'] == 10.0

    def test_iznn_metadata(self):
        neurons = {
            0: IZNeuron(bias=5.0, a=0.02, b=0.20, c=-65.0, d=8.0, inputs=[(-1, 1.0)])
        }
        net = IZNN(neurons, inputs=[-1], outputs=[0])
        data = export_iznn(net, metadata={'generation': 7})
        assert data['metadata']['generation'] == 7
        assert data['network_type'] == 'iznn'
        # Verify IZNN-specific node fields
        output_nodes = [n for n in data['nodes'] if n['type'] == 'output']
        assert output_nodes[0]['a'] == 0.02
        assert output_nodes[0]['d'] == 8.0

    def test_get_neat_version(self):
        """_get_neat_version returns a string (lines 18-20)."""
        v = _get_neat_version()
        assert isinstance(v, str)
        assert len(v) > 0


# ──────────────────────────────────────────────────────────────────────
# InnovationTracker pickle (lines 131, 134, 137-156)
# ──────────────────────────────────────────────────────────────────────

class TestInnovationTrackerPickle:
    def test_repr(self):
        tracker = InnovationTracker(start_number=100)
        r = repr(tracker)
        assert 'InnovationTracker' in r
        assert '100' in r

    def test_pickle_roundtrip(self):
        tracker = InnovationTracker(start_number=50)
        # Assign some innovations so generation_innovations is populated
        tracker.get_innovation_number(-1, 0, 'add_connection')
        tracker.get_innovation_number(-2, 0, 'add_connection')

        data = pickle.dumps(tracker)
        restored = pickle.loads(data)

        assert restored.global_counter == tracker.global_counter
        # Dedup still works: same key returns same number
        orig_num = tracker.get_innovation_number(-1, 0, 'add_connection')
        rest_num = restored.get_innovation_number(-1, 0, 'add_connection')
        assert orig_num == rest_num

    def test_get_current_innovation_number(self):
        tracker = InnovationTracker()
        assert tracker.get_current_innovation_number() == 0
        tracker.get_innovation_number(1, 2, 'add_connection')
        assert tracker.get_current_innovation_number() == 1


# ──────────────────────────────────────────────────────────────────────
# IZNeuron overflow handling (iznn lines 107-110)
# ──────────────────────────────────────────────────────────────────────

class TestIZNNOverflow:
    def test_neuron_overflow_resets_without_spike(self):
        """Overflow during advance resets neuron without producing spike."""
        neuron = IZNeuron(bias=0.0, a=0.02, b=0.2, c=-65.0, d=8.0, inputs=[])
        # Drive voltage to extreme value that will overflow during quadratic update
        neuron.v = 1e154
        neuron.current = 1e154
        neuron.advance(1.0)
        # After overflow reset: v = c, u = b*v
        assert neuron.v == neuron.c
        assert neuron.u == pytest.approx(neuron.b * neuron.c)
        assert neuron.fired == 0.0

    def test_iznn_advance_returns_fired_list(self):
        """IZNN.advance returns list of fired values for output neurons (line 169)."""
        neurons = {
            0: IZNeuron(bias=5.0, a=0.02, b=0.20, c=-65.0, d=8.0, inputs=[(-1, 1.0)])
        }
        net = IZNN(neurons, inputs=[-1], outputs=[0])
        net.set_inputs([1.0])
        result = net.advance(0.05)
        assert isinstance(result, list)
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────
# Gene edge cases (genes.py lines 35, 41-42, 168)
# ──────────────────────────────────────────────────────────────────────

class TestGeneEdgeCases:
    def test_connection_gene_equality_by_innovation(self):
        """Two connection genes with same innovation number are equal (line 165-169)."""
        g1 = DefaultConnectionGene(key=(-1, 0), innovation=42)
        g2 = DefaultConnectionGene(key=(-2, 1), innovation=42)
        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_connection_gene_inequality(self):
        g1 = DefaultConnectionGene(key=(-1, 0), innovation=1)
        g2 = DefaultConnectionGene(key=(-1, 0), innovation=2)
        assert g1 != g2

    def test_connection_gene_not_equal_to_other_type(self):
        g = DefaultConnectionGene(key=(-1, 0), innovation=1)
        assert g != "not a gene"

    def test_node_gene_parse_config_noop(self):
        """BaseGene.parse_config is a no-op (line 35)."""
        # Should not raise
        DefaultNodeGene.parse_config(None, {})

    def test_gene_get_config_params_deprecation(self):
        """Fallback from __gene_attributes__ warns (lines 41-42)."""
        class OldStyleGene(neat.genes.BaseGene):
            __gene_attributes__ = []
            def distance(self, other, config): return 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OldStyleGene.get_config_params()
            assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ──────────────────────────────────────────────────────────────────────
# Reproduction: spawn adjustment and interspecies crossover
# (lines 130, 168, 171, 253, 287, 307-310)
# ──────────────────────────────────────────────────────────────────────

class TestReproductionSpawnAdjust:
    def test_adjust_spawn_adds_to_smaller_species(self):
        """When total < pop_size, extras go to smaller species (line 140-150)."""
        r = DefaultReproduction.__new__(DefaultReproduction)
        # 3 species, sizes 5,5,5 = 15, but we need 18
        result = r._adjust_spawn_exact([5, 5, 5], pop_size=18, min_species_size=2)
        assert sum(result) == 18

    def test_adjust_spawn_removes_from_larger_species(self):
        """When total > pop_size, excess removed from larger species (line 151-165)."""
        r = DefaultReproduction.__new__(DefaultReproduction)
        # Total is 12, need 10 — remove 2 from the largest
        result = r._adjust_spawn_exact([6, 4, 2], pop_size=10, min_species_size=1)
        assert sum(result) == 10
        # Largest species should have shrunk
        assert result[0] <= 6

    def test_adjust_spawn_exact_match(self):
        """No adjustment needed when already matching."""
        r = DefaultReproduction.__new__(DefaultReproduction)
        result = r._adjust_spawn_exact([5, 5], pop_size=10, min_species_size=2)
        assert result == [5, 5]

    def test_adjust_spawn_min_species_conflict(self):
        """Raises when pop_size < num_species * min_species_size (line 130)."""
        r = DefaultReproduction.__new__(DefaultReproduction)
        with pytest.raises(RuntimeError, match="Configuration conflict"):
            r._adjust_spawn_exact([5, 5, 5], pop_size=5, min_species_size=3)


# ──────────────────────────────────────────────────────────────────────
# Genome: output-to-output connection blocked (line 484)
# ──────────────────────────────────────────────────────────────────────

class TestGenomeOutputToOutput:
    def test_no_output_to_output_connections(self):
        """Mutation should never create connections between two output nodes."""
        config = _load_config()
        gc = config.genome_config
        # Only relevant when there are 2+ outputs
        if gc.num_outputs < 2:
            pytest.skip("Need 2+ outputs to test output-to-output block")
        p = neat.Population(config)
        genome = list(p.population.values())[0]
        genome.fitness = 1.0
        # Attempt many add-connection mutations
        for _ in range(200):
            genome.mutate_add_connection(gc)
        # Check no connection goes from output to output
        for (i, o) in genome.connections:
            assert not (i in gc.output_keys and o in gc.output_keys), \
                f"Output-to-output connection found: {i} -> {o}"


# ──────────────────────────────────────────────────────────────────────
# Parallel evaluator: context manager and seed wrapper
# (parallel.py lines 11-15, 43-46)
# ──────────────────────────────────────────────────────────────────────

class TestParallelEvaluator:
    @staticmethod
    def _eval_genome(genome, config):
        return float(genome.key)

    @staticmethod
    def _eval_genome_random(genome, config):
        """Fitness depends on random number, so seed matters."""
        return random.random()

    def test_context_manager(self):
        """ParallelEvaluator can be used as context manager (lines 82-89)."""
        with neat.ParallelEvaluator(2, self._eval_genome) as pe:
            assert pe is not None
        # After exit, pool should be cleaned up
        assert pe._closed

    def test_seeded_evaluation_is_deterministic(self):
        """Same seed + same genome key produces same fitness (lines 43-46)."""
        config = _load_config()
        p = neat.Population(config)
        genomes = list(p.population.items())[:4]

        results1 = {}
        with neat.ParallelEvaluator(2, self._eval_genome_random, seed=42) as pe:
            pe.evaluate(genomes, config)
            for gid, g in genomes:
                results1[gid] = g.fitness

        # Reset and re-evaluate with same seed
        results2 = {}
        with neat.ParallelEvaluator(2, self._eval_genome_random, seed=42) as pe:
            pe.evaluate(genomes, config)
            for gid, g in genomes:
                results2[gid] = g.fitness

        for gid in results1:
            assert results1[gid] == results2[gid], \
                f"Genome {gid}: fitness not reproducible with same seed"

    def test_no_seed_still_works(self):
        """Without seed, evaluation proceeds normally."""
        config = _load_config()
        p = neat.Population(config)
        genomes = list(p.population.items())[:3]
        with neat.ParallelEvaluator(2, self._eval_genome) as pe:
            pe.evaluate(genomes, config)
            for gid, g in genomes:
                assert g.fitness == float(gid)

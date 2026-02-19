import unittest
import e3x
import jax
import jax.numpy as jnp
import jraph

from hfm import utils
from hfm.backbones.utils import build_graph
from hfm.datasets.utils import jraph_center_pos, jraph_get_temperature, jraph_remove_global_rotation_3d, jraph_rotation_augmentation, jraph_stationary


class TestJraphOps(unittest.TestCase):
    def setUp(self):
        rng = jax.random.PRNGKey(0)
        bs = 10
        n_particles = 5
        n_dim = 3

        x = jax.random.normal(rng, (bs, n_particles, n_dim))
        f = jax.random.normal(rng, (bs, n_particles, n_dim))
        p = jax.random.normal(rng, (bs, n_particles, n_dim))
        masses = jax.random.uniform(rng, (bs, n_particles, 1)) + 0.1

        self.data = {"x": x, "f": f, "p": p, "masses": masses}
        self.graph = build_graph(self.data, num_graph=bs, num_node=n_particles, global_properties=())

        return super().setUp()
    
    def test_center_pos(self):
        x_centered = utils.remove_center(self.data["x"])
        graph_centered = jraph_center_pos(self.graph)
        self.assertTrue(jnp.allclose(x_centered.reshape(-1), graph_centered.nodes["x"].reshape(-1)[:-3]))

    def test_get_temperature(self):
        for zero_drift in [True, False]:
            for zero_rot in [True, False]:
                with self.subTest(zero_drift=zero_drift, 
                                  zero_rot=zero_rot):

                    temp = utils.get_temperature(self.data["p"], 
                                                 self.data["masses"], 
                                                 n_dim=3, 
                                                 zero_drift=zero_drift, 
                                                 zero_rot=zero_rot)
                    temp_jraph = jraph_get_temperature(self.graph, 
                                                       n_dim=3, 
                                                       zero_drift=zero_drift, 
                                                       zero_rot=zero_rot)
                    self.assertTrue(jnp.allclose(temp.reshape(-1), 
                                                 temp_jraph.reshape(-1)[:-1]))

    def test_jraph_stationary(self):
        for force_temperature in [True, False]:
            for zero_drift in [True, False]:
                for zero_rot in [True, False]:
                    with self.subTest(force_temperature=force_temperature, 
                                      zero_drift=zero_drift, 
                                      zero_rot=zero_rot):
                        
                        p_stationary = utils.stationary(self.data["p"], 
                                                        self.data["masses"], 
                                                        force_temperature=force_temperature, 
                                                        zero_drift=zero_drift, zero_rot=zero_rot)
                        p_stationary_jraph = jraph_stationary(self.graph, 
                                                              force_temperature=force_temperature, 
                                                              zero_drift=zero_drift, 
                                                              zero_rot=zero_rot)
                        self.assertTrue(jnp.allclose(p_stationary.reshape(-1), 
                                                     p_stationary_jraph.nodes["p"].reshape(-1)[:-3]))
    
    def test_jraph_zero_rotation(self):
        for force_temperature in [True, False]:
            for zero_drift in [True, False]:
                for zero_rot in [True, False]:
                    with self.subTest(force_temperature=force_temperature, 
                                      zero_drift=zero_drift, 
                                      zero_rot=zero_rot):
                        
                        p_zero_rot = utils.zero_rotation(self.data["x"], 
                                                         self.data["p"], 
                                                         self.data["masses"],
                                                         force_temperature=force_temperature, 
                                                         zero_drift=zero_drift, 
                                                         zero_rot=zero_rot)
                        p_zero_rot_jraph = jraph_remove_global_rotation_3d(self.graph, 
                                                                     force_temperature=force_temperature,
                                                                     zero_drift=zero_drift, 
                                                                     zero_rot=zero_rot)
                        self.assertTrue(jnp.allclose(p_zero_rot.reshape(-1), 
                                                     p_zero_rot_jraph.nodes["p"].reshape(-1)[:-3]))

    def test_jraph_random_rotation(self):
        for ndim in [1, 2, 3]:
            with self.subTest(ndim=ndim):
                rng = jax.random.PRNGKey(42)
                test_graph = self.graph._replace(nodes={k: v[..., :ndim] for k, v in self.graph.nodes.items()})
                test_data = {k: v[..., :ndim] for k, v in self.data.items()}

                self.assertEqual(test_graph.nodes["x"].shape[-1], ndim)
                self.assertEqual(test_data["x"].shape[-1], ndim)
                bs = test_data["x"].shape[0]

                if ndim == 3:
                    rot = e3x.so3.random_rotation(key=rng, num=bs)
                elif ndim == 2:
                    rot = utils.random_rotation_2d(key=rng, num=bs)
                elif ndim == 1:
                    rot = jnp.ones((bs, 1, 1))

                x = jnp.einsum('bij,bjk->bik', test_data["x"], rot)
                f = jnp.einsum('bij,bjk->bik', test_data["f"], rot)
                p = jnp.einsum('bij,bjk->bik', test_data["p"], rot)

                jraph_augmented = jraph_rotation_augmentation(rng, test_graph, rotation_props=["x", "f", "p"])

                self.assertTrue(jnp.allclose(x.reshape(-1), jraph_augmented.nodes["x"].reshape(-1)[:-ndim]))
                self.assertTrue(jnp.allclose(f.reshape(-1), jraph_augmented.nodes["f"].reshape(-1)[:-ndim]))
                self.assertTrue(jnp.allclose(p.reshape(-1), jraph_augmented.nodes["p"].reshape(-1)[:-ndim]))

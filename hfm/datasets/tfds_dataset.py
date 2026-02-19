from dataclasses import dataclass
from functools import partial
import os
import queue
import threading
import time
import jraph
import numpy as np
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import ase


@dataclass
class WorkerConfig:
    data_dir: str
    split: str
    cutoff: float
    max_num_graphs: int
    num_atoms: int
    shuffle_seed: int
    debug: bool # whether to take a subset of data
    n_workers: int
    worker_idx: int
    skip_loading_prefetch: bool

    
class StopToken:
    pass


class TfdsDataset:
    STOP_TOKEN = StopToken()
    N_DEBUG = 10_000

    def __init__(self, data_dir, cutoff, num_atoms_mean, n_proc=4, skip_loading_prefetch=False, debug=False):
        self.data_dir = data_dir
        self.cutoff = cutoff
        self.n_proc = n_proc
        self.skip_loading_prefetch = skip_loading_prefetch
        self.num_atoms_mean = num_atoms_mean
        self.debug = debug

        if self.n_proc > 0:
            # multithread stuff
            ctx = get_context("spawn")
            self.manager = ctx.Manager()
            self.executor = ProcessPoolExecutor(max_workers=self.n_proc, mp_context=ctx)

            self.stop_event = threading.Event()
            # Start the monitor thread
            self.monitor_thread = threading.Thread(target=TfdsDataset._monitor_parent_children, args=(1.0, self.stop_event))
            self.monitor_thread.start()

    @staticmethod
    def compute_edges_tf(
        positions,
        cutoff: float
    ):
        num_atoms = tf.shape(positions)[0]
        displacements = positions[None, :, :] - positions[:, None, :]
        distances = tf.norm(displacements, axis=-1)
        mask = ~tf.eye(num_atoms, dtype=tf.bool)  # Get rid of self-connections.
        keep_edges = tf.where((distances < cutoff) & mask)
        centers = tf.cast(keep_edges[:, 0], dtype=tf.int32)  # center indices
        others = tf.cast(keep_edges[:, 1], dtype=tf.int32)  # neighbor indices
        return centers, others

    @staticmethod
    def create_graph_tuples(element, cutoff):
        globals_dict = dict()
        nodes_dict = dict()

        atomic_numbers = tf.cast(element['atomic_numbers'], dtype=tf.int32)
        positions = tf.cast(element['positions'], dtype=tf.float32)
        forces = tf.cast(element['forces'], dtype=tf.float32)
        energy = tf.cast(element['energy'], dtype=tf.float32)

        nodes_dict['atomic_numbers'] = tf.reshape(atomic_numbers, (-1, 1))
        nodes_dict['x'] = positions
        nodes_dict['f'] = forces
        globals_dict['Epot'] = tf.reshape(energy, (-1, 1))

        centers, others = TfdsDataset.compute_edges_tf(
            positions=positions,
            cutoff=cutoff
        )

        num_nodes = tf.shape(atomic_numbers)[0]
        num_edges = tf.shape(centers)[0]

        return jraph.GraphsTuple(
            n_node=tf.reshape(num_nodes, (1,)),
            n_edge=tf.reshape(num_edges, (1,)),
            receivers=centers,
            senders=others,
            nodes=nodes_dict,
            globals=globals_dict,
            edges={},
        )

    @staticmethod
    def _preprocess(dataset, cutoff, num_atoms, max_num_graphs):
        cgt = partial(TfdsDataset.create_graph_tuples, cutoff=cutoff)

        # Remove atom-wise energy bias
        shifts = tf.constant([0,-16.480896468322097,0,0,0,0,-1035.229870597111,-1488.1736875491576,-2045.351086366964,0,0,0,0,0,0,0,-10832.697554632095,-12520.733604908672], dtype=tf.float64)
        
        def compute_shifted_energy(sample):
            atn = tf.cast(sample["atomic_numbers"], tf.int32)
            # Gather per-atom shifts using atomic numbers as indices
            energy_shift = tf.gather(shifts, atn)
            # Subtract the sum of per-atom shifts from total energy
            sample["energy"] = sample["energy"] - tf.reduce_sum(energy_shift)
            return sample

        dataset = dataset.map(
            compute_shifted_energy,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).map(cgt,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).shuffle(
            buffer_size=10_000,
            reshuffle_each_iteration=True,  # we won't have more than one iteration so probably doesn't matter
        ).prefetch(tf.data.AUTOTUNE)

        batched_dataset = jraph.dynamically_batch(
            dataset.as_numpy_iterator(),
            n_graph=max_num_graphs,
            n_node=(max_num_graphs - 1) * num_atoms + 1,
            n_edge=(max_num_graphs - 1) * num_atoms * num_atoms + 1)
        
        for batch in batched_dataset:
            masses = ase.Atoms(batch.nodes["atomic_numbers"].reshape(-1)).get_masses().reshape(-1, 1)
            batch.nodes['masses'] = masses
            yield batch

    @staticmethod
    def _safe_put(queue, item):
        try:
            queue.put(item)
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            print(f"[!] Queue closed before item could be put: {e}")
        except Exception as e:
            print(f"[!] Unexpected error putting item to queue: {e}")

    @staticmethod
    def _worker(config, output_queue):
        """Worker function that loads data for the given indices and puts it into the queue."""

        try:
            # We need a deterministic shuffle seed, s.t. workers shuffle files in the same way
            read_config = tfds.ReadConfig(
                shuffle_seed=config.shuffle_seed,
                skip_prefetch=config.skip_loading_prefetch
            )

            # Note that tfds automatically pre-fetches after reading
            # This might be suboptimal if we prefetch later and we can try to disable it
            builder = tfds.builder_from_directory(config.data_dir)
            dataset = builder.as_dataset(split=config.split, shuffle_files=True, read_config=read_config)

            if config.debug:
                dataset = dataset.take(TfdsDataset.N_DEBUG)

            dataset = dataset.shard(num_shards=config.n_workers, index=config.worker_idx)

            for batch in TfdsDataset._preprocess(dataset, 
                                                 cutoff=config.cutoff,
                                                 num_atoms=config.num_atoms, 
                                                 max_num_graphs=config.max_num_graphs):                
                output_queue.put(batch)
        except Exception as e:
            print(f"[!] Error in worker {config.worker_idx}: {e}")
        finally:
            # Finished processing data, always put a stop token
            TfdsDataset._safe_put(output_queue, TfdsDataset.STOP_TOKEN)
    
    @staticmethod
    def _monitor_parent_children(interval=60.0, stop_event=None):
        """Monitor all child processes of the current process."""
        parent = psutil.Process(os.getpid())
        print("[Monitor] Started monitoring subprocesses...")

        while not stop_event.is_set():
            children = parent.children(recursive=True)
            mem = parent.memory_info().rss / (1024 * 1024 * 1024)
            for child in children:
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        mem += child.memory_info().rss / (1024 * 1024 * 1024)  # GB
                except psutil.NoSuchProcess:
                    pass
            if wandb.run is not None:
                wandb.log({"total_memory_GB": mem})
            time.sleep(interval)

        print("[Monitor] Stopped monitoring.")

    def _generator(self, batch_size, split):
        """Generator reads data from the queue."""
        shuffle_seed = np.random.randint(0, 2**31 - 1) # different shuffle seed for each epoch
        output_queue = self.manager.Queue(maxsize=10) # cache maximum of 100 batches, use a fresh queue to collect results

        # processes = []
        for i in range(self.n_proc):
            config = WorkerConfig(
                data_dir=self.data_dir,
                split=split,
                cutoff=self.cutoff,
                max_num_graphs=batch_size,
                num_atoms=self.num_atoms_mean,
                shuffle_seed=shuffle_seed,
                debug=self.debug,
                n_workers=self.n_proc,
                worker_idx=i,
                skip_loading_prefetch=self.skip_loading_prefetch
            )
            self.executor.submit(TfdsDataset._worker, config, output_queue)

        n_stop_token = 0
        ctr = 0
        while True:
            ctr += 1

            try:
                batch = output_queue.get(timeout=3)
            except queue.Empty:
                # Timeout expired, try again
                continue
            except Exception as e:
                print(f"[!] Unexpected error retrieving item from queue: {e}")
                break

            if ctr % 100 == 0:
                size = output_queue.qsize()
                if wandb.run is not None:
                    wandb.log({"queue_size": size})

            if isinstance(batch, StopToken):
                n_stop_token += 1

                # wait for all processes to finish
                if n_stop_token == self.n_proc:
                    break
            else:
                yield batch

        del output_queue

    def _generator_sync(self, batch_size, split):
        builder = tfds.builder_from_directory(self.data_dir)
        dataset = builder.as_dataset(split=split, shuffle_files=True)

        if self.debug:
            dataset = dataset.take(TfdsDataset.N_DEBUG)

        for batch in self._preprocess(dataset, 
                                 cutoff=self.cutoff,
                                 num_atoms=self.num_atoms_mean, 
                                 max_num_graphs=batch_size):
            yield batch

    def get_len(self, split):
        builder = tfds.builder_from_directory(self.data_dir)
        dataset = builder.as_dataset(split=split, shuffle_files=True)
        num_examples = len(dataset)

        if self.debug:
            num_examples = min(TfdsDataset.N_DEBUG, num_examples)

        return num_examples

    def next_epoch(self, batch_size, split):
        # Loads the data for ONE epoch
        if self.n_proc > 0:
            return self._generator(batch_size, split)
        else:
            return self._generator_sync(batch_size)

    def get_example_batch(self, batch_size, split):
        for sample in self._generator_sync(batch_size, split):
            return sample

    def shutdown(self):
        if self.n_proc > 0:
            print("Shutting down monitor thread...")
            self.stop_event.set()
            self.monitor_thread.join()
            print("Shutting down manager...")
            self.manager.shutdown()
            time.sleep(5)  # Give some time for the workers to finish
            print("Shutting down executor...")
            self.executor.shutdown(wait=False, cancel_futures=True)

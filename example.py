import argparse

import pickle
import gzip

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer


def parse_arguments() -> argparse.Namespace:
    """ This function parses the arguments.

    Returns:
        argparse.Namespace: The command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', required=True)
    parser.add_argument('--input_NP_Score_path', required=True)
    parser.add_argument('--input_SA_Score_path', required=True)
    parser.add_argument('--output_log_path', required=True)
    parser.add_argument('--output_model_dir', required=True)
    parser.add_argument('--validation_metrics', required=False, default='validity,sas')
    parser.add_argument('--num_epochs', required=False, default=100)
    parser.add_argument('--save_frequency', required=False, default=1)

    args = parser.parse_args()
    return args


args = parse_arguments()

# The method was introduced in the paper titled
# "MolGAN: An Implicit Generative Model for Small Molecular Graphs."
# You can access comprehensive information about this paper
# at https://arxiv.org/pdf/1805.11973.pdf. The specific numerical values
# for batch size, decoder, and discriminator parameters are derived from this paper.
batch_dim = 128
la = 1
dropout = 0
n_critic = 5
metric = args.validation_metrics
n_samples = 5000
z_dim = 8
epochs = int(args.num_epochs)
save_every = int(args.save_frequency) # May lead to errors if left as None


# The unpicking of score files is moved from "molecular_metrics.py"
# to here because the path to score files is an argument of this script.
NP_model = pickle.load(gzip.open(args.input_NP_Score_path))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open(args.input_SA_Score_path)) for j in range(1, len(i))}


data = SparseMolecularDataset()
data.load(args.input_data_path)

steps = (len(data) // batch_dim)


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    a = [optimizer.train_step_G] if i % n_critic == 0 else [optimizer.train_step_D]
    b = [optimizer.train_step_V] if i % n_critic == 0 and la < 1 else []
    return a + b


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
    embeddings = model.sample_z(batch_dim)

    if la < 1:

        if i % n_critic == 0:
            rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

            rewardF = reward(mols)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}

        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}
    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.training: True,
                     model.dropout_rate: dropout,
                     optimizer.la: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def test_fetch_dict(model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def test_feed_dict(model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def reward(mols):
    rr = 1.
    mols_metrics = MolecularMetrics()   
    for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

        if m == 'np':
            rr *= mols_metrics.natural_product_scores(mols, NP_model, norm=True)
        elif m == 'logp':
            rr *= mols_metrics.water_octanol_partition_coefficient_scores(mols, norm=True)
        elif m == 'sas':
            rr *= mols_metrics.synthetic_accessibility_score_scores(mols, SA_model, norm=True)
        elif m == 'qed':
            rr *= mols_metrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
        elif m == 'novelty':
            rr *= mols_metrics.novel_scores(mols, data)
        elif m == 'dc':
            rr *= mols_metrics.drugcandidate_scores(mols, SA_model, data)
        elif m == 'unique':
            rr *= mols_metrics.unique_scores(mols)
        elif m == 'diversity':
            rr *= mols_metrics.diversity_scores(mols, data)
        elif m == 'validity':
            rr *= mols_metrics.valid_scores(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
    
    m0, m1 = all_scores(mols, data, NP_model, SA_model, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


def _test_update(model, optimizer, batch_dim, test_batch):
    mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, NP_model, SA_model, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


# model
model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((128, 64), 128, (128, 64)),
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=False,
                      hard_gumbel_softmax=False,
                      batch_discriminator=False)

# optimizer
optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)

# session
session = tf.Session()
session.run(tf.global_variables_initializer())

# trainer
trainer = Trainer(model, optimizer, session)

print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

trainer.train(batch_dim=batch_dim,
              epochs=epochs,
              steps=steps,
              train_fetch_dict=train_fetch_dict,
              train_feed_dict=train_feed_dict,
              eval_fetch_dict=eval_fetch_dict,
              eval_feed_dict=eval_feed_dict,
              test_fetch_dict=test_fetch_dict,
              test_feed_dict=test_feed_dict,
              save_every=save_every,
              directory=args.output_model_dir, # here users need to first create and then specify a folder where to save the model
              _eval_update=_eval_update,
              _test_update=_test_update,
              output_log_path=args.output_log_path)

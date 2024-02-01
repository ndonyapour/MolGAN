import argparse

import pickle
import gzip

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.sanitize_ligand import is_valid_ligand
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer
import rdkit
from rdkit import Chem



def parse_arguments() -> argparse.Namespace:
    """ This function parses the arguments.

    Returns:
        argparse.Namespace: The command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', required=True)
    parser.add_argument('--input_NP_Score_path', required=True)
    parser.add_argument('--input_SA_Score_path', required=True)
    parser.add_argument('--input_model_dir', required=True)
    parser.add_argument('--output_log_path', required=True)
    parser.add_argument('--output_sdf_path', required=True)
    parser.add_argument('--num_samples', required=False, default=1000)
    args = parser.parse_args()
    return args

args = parse_arguments()

# The method was introduced in the paper titled
# "MolGAN: An Implicit Generative Model for Small Molecular Graphs."
# You can access comprehensive information about this paper
# at https://arxiv.org/pdf/1805.11973.pdf. The specific numerical values
# for batch size, decoder, and discriminator parameters are derived from this paper.
z_dim = 8
batch_dim = 128

n_samples = int(args.num_samples)

NP_model = pickle.load(gzip.open(args.input_NP_Score_path))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open(args.input_SA_Score_path)) for j in range(1, len(i))}

data = SparseMolecularDataset()
data.load(args.input_data_path)

model_dir = args.input_model_dir
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

optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)

session = tf.Session()
session.run(tf.global_variables_initializer())

trainer = Trainer(model, optimizer, session)
trainer.load(model_dir)

# Generate new molecules
mols = samples(data, model, session, model.sample_z(n_samples), sample=True)

# Evaluate the generated molecules
m0, m1 = all_scores(mols, data, NP_model, SA_model, norm=True)

# Refine the selection to include valid molecules
# and filter out molecules with disconnected components
valid_mols = list(filter(lambda mol: mol is not None and
                         len(Chem.rdmolops.GetMolFrags(mol))==1, mols))

# We exclude the generated small molecules that are not kekulizable.
num_notkekulizable = 0
with Chem.SDWriter(args.output_sdf_path) as w:
    for mol in valid_mols:
        if Chem.SanitizeMol(mol, catchErrors=True) == 0:
            w.write(mol)
        else:
            num_notkekulizable += 1

# Save the valid molecules into an SDF file
with open(args.output_log_path, mode='w', encoding='utf-8') as wfile:
    wfile.write(str(m1))
    wfile.write(f'\n{len(valid_mols)}'
                f'from {len(mols) - num_notkekulizable} generated molecules are valid')
    
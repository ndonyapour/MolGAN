
import os.path as osp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer
from rdkit import Chem
  

z_dim = 8
batch_dim = 128
n_samples = 1000

data = SparseMolecularDataset()
data.load('data/gdb9_9nodes.sparsedataset')

# load the model
model_dir = './trained_models'
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
trainer.load(model_dir)

# Generate new molecules
mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
 
# evaluate the generated molecules
_, m1 = all_scores(mols, data, norm=True)
print(m1)

# filter the valid molecules
valid_mols = list(filter(lambda x: x is not None, mols))

with Chem.SDWriter(osp.join(model_dir, 'generated.sdf')) as w:
  for m in valid_mols:
        w.write(m)

print(f'{len(valid_mols)} from {len(mols)} generated molecules are valid')
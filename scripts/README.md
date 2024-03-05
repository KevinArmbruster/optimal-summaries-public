
Helpful commands

find /workdir/optimal-summaries-public/_models2 -type d -name "weight_magnitude" -print
find /workdir/optimal-summaries-public/_models2 -type d -name "weight_magnitude" -exec rm -rf {} +
find /workdir/optimal-summaries-public/_models2 -type d -name "gradient_magnitude" -exec rm -rf {} +
find /workdir/optimal-summaries-public/_models2 -type d -name "weight_gradient_magnitude" -exec rm -rf {} +
find /workdir/optimal-summaries-public/_models2 -type d -name "sparse_learning" -exec rm -rf {} +

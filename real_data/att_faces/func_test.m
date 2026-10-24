rng(2);
tic
clusterKMfull = kmeans(X.', 40, ...
    'Replicates', 20, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSec = toc;
fprintf('solve time of kmeans cluter on full data: %.2f s\n', fullKmSec);

Ckmfull = accumarray([labels, clusterKMfull], 1, [40, 40]);
countsKMfull = accumarray(clusterKMfull, 1, [40, 1]);
purityKMfull = sum(max(Ckmfull, [], 1)) / numel(labels);

fprintf('purity of kmeans cluster on full data: %.5f\n', purityKMfull);
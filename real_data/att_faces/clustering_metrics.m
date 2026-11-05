function [purity, accuracy] = clustering_metrics(labels, cluster, numClasses, numClusters)
%CLUSTERING_METRICS Purity and one-to-one label-matched clustering accuracy.
% Rows of C are true classes; columns are predicted clusters.

    C = accumarray([labels(:), cluster(:)], 1, [numClasses, numClusters]);
    purity = sum(max(C, [], 1)) / numel(labels);

    % Ding et al.: maximize the confusion-matrix diagonal after a one-to-one
    % permutation of cluster labels. A large unassignment cost forces every
    % class and cluster to participate, including empty predicted clusters.
    assert(numClasses == numClusters, ...
        'Accuracy requires equal numbers of classes and clusters.');
    cost = max(C(:)) - C;
    pairs = matchpairs(cost, max(cost(:)) + 1);
    assert(size(pairs, 1) == numClasses, ...
        'Expected a complete one-to-one class/cluster matching.');
    accuracy = sum(C(sub2ind(size(C), pairs(:,1), pairs(:,2)))) ...
        / numel(labels);
    assert(accuracy <= purity + 1e-12);
end

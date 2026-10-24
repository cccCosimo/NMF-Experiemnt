Z = Gfull .* vecnorm(Ffull, 2, 1);
Z = Z ./ max(vecnorm(Z, 2, 2), eps);

rng(2);
tic
clusterKM = kmeans(Z, 40, ...
    'Replicates', 20, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
secondsKM = toc;

Ckm = accumarray([labels, clusterKM], 1, [40, 40]);
countsKM = accumarray(clusterKM, 1, [40, 1]);
purityKM = sum(max(Ckm, [], 1)) / numel(labels);

fprintf('k-means Purity: %.4f\n', purityKM);
fprintf('k-means solve time: %.2f s\n', secondsKM);
fprintf('nonempty: %d / 40，cluster number scale: %d–%d\n', ...
    nnz(countsKM), min(countsKM), max(countsKM));

[~, order] = sort(countsKM, 'descend');
shownClusters = order(1:6);

rng(3);  % fix which images to view after clustering
figure;
for row = 1:6
    members = find(clusterKM == shownClusters(row));
    chosen = members(randperm(numel(members), min(6, numel(members))));

    for col = 1:numel(chosen)
        subplot(6, 6, (row-1)*6 + col);
        imagesc(reshape(X(:,chosen(col)), 112, 92));
        colormap gray;
        axis image off;
        title(sprintf('ID %d', labels(chosen(col))));
        if col == 1
            ylabel(sprintf('cluster %d\n %d images', ...
                shownClusters(row), numel(members)));
        end
    end
end
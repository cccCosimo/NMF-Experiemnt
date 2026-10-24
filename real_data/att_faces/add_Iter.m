r = 40;
maxIter = 150;
rng(1);
Ginit = rand(size(X, 2), r);

tic;
[Ffull, Gfull, history] = seminmf_hals(X, Ginit, maxIter);
solveSeconds = toc;

curve = sqrt(2 * history) / norm(X, 'fro');
fprintf('error No.50 %.6f, No.100 %.6f, No.150 %.6f\n',...
    curve(51), curve(101), curve(151));
fprintf('solve time of 150 iterations: %.2f s\n', solveSeconds);

% dismiss the effect of scale ambiguity to the maxmum element indicator
scores = Gfull .* vecnorm(Ffull, 2, 1);
[~, cluster] = max(scores, [], 2);

counts = accumarray(cluster, 1, [r, 1]);
C = accumarray([labels, cluster], 1, [r, r]);
purity = sum(max(C, [], 1)) / numel(labels);

fprintf('nonempty cluter: %d / %d\n', nnz(counts), r);
fprintf('element numbers of each cluter: maxmum %d, minimul %d\n', min(counts), max(counts));
fprintf('Purity: %.4f\n', purity);

figure;
imagesc(C);
xlabel('cluster indices');
ylabel('real indices');
title('ORL identity correspondence to cluster');
colorbar




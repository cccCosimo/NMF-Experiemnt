% integrate all the parts of a complete experiment.
% fix rank = 40, skitch_size = 240 and iterations = 150.
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
dataRoot = fullfile(scriptDir, 'att_faces_data');

height = 112;
width = 92;           % 112 * 92 pixels per image
nPeople = 40;
nPerPerson = 10;
n = nPeople * nPerPerson;    % 10 images per person

% vectorize image 112 * 92 matrix
% each column of X represents a image
X = zeros(height * width, n);
labels = zeros(n, 1);

%load faces data
for person = 1:nPeople
    for imageNo = 1:nPerPerson
        col = (person - 1) * nPerPerson + imageNo;
        file = fullfile(dataRoot, sprintf('s%d', person),...
                        sprintf('%d.pgm', imageNo));
        
        img = imread(file);
        assert(isequal(size(img), [height, width]), ...
            'image does not conform to the formal size', file);
        
        X(:, col) = double(img(:)) / 255;
        labels(col) = person;
    end
end
% assert if somethiing went wrong
assert(isequal(size(X), [height*width, nPeople*nPerPerson]));
assert(all(isfinite(X(:))));
assert(all(X(:) >= 0 & X(:) <= 1));
assert(numel(unique(labels)) == 40);

fprintf('X: %d * %d\n', size(X, 1), size(X,2));
fprintf('pixel scale: [%.3f, %.3f]\n', min(X(:)), max(X(:)));

% diaplay few samples
figure;
for j = 1:10
    subplot(2, 5, j);
    imagesc(reshape(X(:, j), height, width));
    colormap gray;
    axis image off;
    title(sprintf('Person %d, image %d', labels(j), j));
end

addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/synthetic_data');

r = 40;
maxIter = 150;
normX = norm(X, 'fro');
rng(1);
Ginit = rand(size(X, 2), r);

tic;
[Ffull, Gfull, history] = seminmf_hals(X, Ginit, maxIter);
fullSolveSeconds = toc;

curve = sqrt(2 * history) / normX;
fullError = norm(X - Ffull*Gfull.', 'fro') / normX;
assert(isequal(size(Ffull), [size(X,1), r]));
assert(isequal(size(Gfull), [size(X,2), r]));
assert(all(isfinite(history)) && all(Gfull(:) >= 0));
assert(abs(curve(end) - fullError) < 1e-10);
fprintf('error No.50 %.6f, No.100 %.6f, No.150 %.6f\n',...
    curve(51), curve(101), curve(151));
fprintf('full-data relative error: %.6f\n', fullError);
fprintf('full-data solve time: %.4f s\n', fullSolveSeconds);

% Clustering time includes construction and normalization of the coefficients.
fullClusterStart = tic;
Z = Gfull .* vecnorm(Ffull, 2, 1);
Z = Z ./ max(vecnorm(Z, 2, 2), eps);

rng(2);
clusterKM = kmeans(Z, 40, ...
    'Replicates', 20, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSeconds = toc(fullClusterStart);

Ckm = accumarray([labels, clusterKM], 1, [40, 40]);
countsKM = accumarray(clusterKM, 1, [40, 1]);
[purityKM, accuracyKM] = clustering_metrics(labels, clusterKM, nPeople, r);

fprintf('k-means Purity: %.4f\n', purityKM);
fprintf('k-means matched Accuracy: %.4f\n', accuracyKM);
fprintf('full-data clustering time: %.4f s\n', fullKmSeconds);
fprintf('full-data factorization + clustering: %.4f s\n', ...
    fullSolveSeconds + fullKmSeconds);
fprintf('nonempty: %d / 40，cluster number scale: %d–%d\n', ...
    nnz(countsKM), min(countsKM), max(countsKM));

% Diagnostic: assign each image to its largest scale-adjusted coefficient.
% This is evaluated separately from the timed k-means clustering pipeline.
maxRuleScores = Gfull .* vecnorm(Ffull, 2, 1);
[~, maxRuleCluster] = max(maxRuleScores, [], 2);
maxRuleCounts = accumarray(maxRuleCluster, 1, [r, 1]);
maxRuleC = accumarray([labels, maxRuleCluster], 1, [nPeople, r]);
[maxRulePurity, maxRuleAccuracy] = ...
    clustering_metrics(labels, maxRuleCluster, nPeople, r);

fprintf('full-data maximum-coefficient Purity: %.4f\n', maxRulePurity);
fprintf('full-data maximum-coefficient matched Accuracy: %.4f\n', maxRuleAccuracy);
fprintf('maximum-coefficient nonempty clusters: %d / %d, size range: %d-%d\n', ...
    nnz(maxRuleCounts), r, min(maxRuleCounts), max(maxRuleCounts));

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

t = 240;

% Use the same initial G as above and a fixed sketch seed.
rng(11);
Omega = randn(size(X,2), t);

totalStart = tic;

stageStart = tic;
Y = X * Omega;
[Q, ~] = qr(Y, 0);
B = Q.' * X;
sketchSeconds = toc(stageStart);

stageStart = tic;
[Fs, Gs, historyS] = seminmf_hals(B, Ginit, maxIter);
sketchSolveSeconds = toc(stageStart);

stageStart = tic;
Fsketch = Q * Fs;
recoverySeconds = toc(stageStart);

sketchFactorizationSeconds = toc(totalStart);

projectionError = norm(X - Q*B, 'fro') / normX;
smallError = norm(B - Fs*Gs.', 'fro') / normX;
sketchError = norm(X - Fsketch*Gs.', 'fro') / normX;
identityGap = abs(sketchError^2 - projectionError^2 - smallError^2);

% The same clustering rules and k-means settings were used as in the full-data experiment
sketchClusterStart = tic;
Zs = Gs .* vecnorm(Fsketch, 2, 1);
Zs = Zs ./ max(vecnorm(Zs, 2, 2), eps);
rng(2);
clusterSketch = kmeans(Zs, 40, ...
    'Replicates', 20, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
sketchKmSeconds = toc(sketchClusterStart);

Csketch = accumarray([labels, clusterSketch], 1, [40, 40]);
[puritySketch, accuracySketch] = ...
    clustering_metrics(labels, clusterSketch, nPeople, r);

fprintf('Projection/Small spaces/Original space error：%.6f / %.6f / %.6f\n', ...
    projectionError, smallError, sketchError);
fprintf('Error identity deviation：%.2e\n', identityGap);
fprintf('Compress/Solve/Restore/Factorization time: %.4f / %.4f / %.4f / %.4f s\n', ...
    sketchSeconds, sketchSolveSeconds, recoverySeconds, sketchFactorizationSeconds);
fprintf('compressed clustering time: %.4f s\n', sketchKmSeconds);
fprintf('compressed factorization + clustering: %.4f s\n', ...
    sketchFactorizationSeconds + sketchKmSeconds);
fprintf('compressed Purity：%.4f\n', puritySketch);
fprintf('compressed matched Accuracy: %.4f\n', accuracySketch);

assert(identityGap < 1e-10);

rng(2);
tic
clusterKMfull = kmeans(X.', 40, ...
    'Replicates', 20, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSec = toc;
fprintf('solve time of kmeans cluter on full data: %.2f s\n', fullKmSec);

Ckmfull = accumarray([labels, clusterKMfull], 1, [40, 40]);
countsKMfull = accumarray(clusterKMfull, 1, [40, 1]);
[purityKMfull, accuracyKMfull] = ...
    clustering_metrics(labels, clusterKMfull, nPeople, r);

fprintf('purity of kmeans cluster on full data: %.5f\n', purityKMfull);
fprintf('matched Accuracy of raw-pixel k-means: %.5f\n', accuracyKMfull);

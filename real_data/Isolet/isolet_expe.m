% isolet data contains features range from -1 to 1, in other words, it is
% a mixed sign data set. The data matrix's dimention would be 617 * 7797.
% each column represents an sample. therefore, it contains less features 
% but much more samples than those previous data set. we could test our
% method's performance on this type of problem

% load data
dataDir = '/Users/huangjiahao/CNMF_with_compression/Experiments/real_data/Isolet/isolet';

A = readmatrix(fullfile(dataDir, 'isolet1+2+3+4.data'), ...
    'FileType', 'text', 'Delimiter', ',');
B = readmatrix(fullfile(dataDir, 'isolet5.data'), ...
    'FileType', 'text', 'Delimiter', ',');

assert(isequal(size(A), [6238, 618]));
assert(isequal(size(B), [1559, 618]));

data = [A; B];
X = data(:, 1:617).';
labels = data(:, 618);

assert(isequal(size(X), [617, 7797]));
assert(all(isfinite(X(:))) && all(isfinite(labels)));
assert(all(labels == fix(labels) & labels >= 1 & labels <= 26));

fprintf('X: %d × %d\n', size(X, 1), size(X, 2));
fprintf('feature range: [%.4f, %.4f]\n', min(X(:)), max(X(:)));
fprintf('classes: %d\n', numel(unique(labels)));

addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/synthetic_data');
addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/real_data/att_faces');

r = 26; % matrix rank or actuall cluster number
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
clusterKM = kmeans(Z, 26, ...
    'Replicates', 26, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSeconds = toc(fullClusterStart);

Ckm = accumarray([labels, clusterKM], 1, [26, 26]);
countsKM = accumarray(clusterKM, 1, [26, 1]);
[purityKM, accuracyKM] = clustering_metrics(labels, clusterKM,r, r);

fprintf('k-means Purity: %.4f\n', purityKM);
fprintf('k-means matched Accuracy: %.4f\n', accuracyKM);
fprintf('full-data clustering time: %.4f s\n', fullKmSeconds);
fprintf('full-data factorization + clustering: %.4f s\n', ...
    fullSolveSeconds + fullKmSeconds);
fprintf('nonempty: %d / 26，cluster number scale: %d–%d\n', ...
    nnz(countsKM), min(countsKM), max(countsKM));

% Diagnostic: assign each sample to its largest scale-adjusted coefficient.
% This is evaluated separately from the timed k-means clustering pipeline.
maxRuleScores = Gfull .* vecnorm(Ffull, 2, 1);
[~, maxRuleCluster] = max(maxRuleScores, [], 2);
maxRuleCounts = accumarray(maxRuleCluster, 1, [r, 1]);
maxRuleC = accumarray([labels, maxRuleCluster], 1, [r, r]);
[maxRulePurity, maxRuleAccuracy] = ...
    clustering_metrics(labels, maxRuleCluster,r, r);

fprintf('full-data maximum-coefficient Purity: %.4f\n', maxRulePurity);
fprintf('full-data maximum-coefficient matched Accuracy: %.4f\n', maxRuleAccuracy);
fprintf('maximum-coefficient nonempty clusters: %d / %d, size range: %d-%d\n', ...
    nnz(maxRuleCounts), r, min(maxRuleCounts), max(maxRuleCounts));

t = 120; % sketch size

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
clusterSketch = kmeans(Zs, 26, ...
    'Replicates', 26, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
sketchKmSeconds = toc(sketchClusterStart);

Csketch = accumarray([labels, clusterSketch], 1, [26, 26]);
[puritySketch, accuracySketch] = ...
    clustering_metrics(labels, clusterSketch,r, r);

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
clusterKMfull = kmeans(X.', 26, ...
    'Replicates', 26, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSec = toc;
fprintf('solve time of kmeans cluter on full data: %.2f s\n', fullKmSec);

Ckmfull = accumarray([labels, clusterKMfull], 1, [26, 26]);
countsKMfull = accumarray(clusterKMfull, 1, [26, 1]);
[purityKMfull, accuracyKMfull] = ...
    clustering_metrics(labels, clusterKMfull,r, r);

fprintf('purity of kmeans cluster on full data: %.5f\n', purityKMfull);
fprintf('matched Accuracy of raw-data k-means: %.5f\n', accuracyKMfull);
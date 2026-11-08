% like Isolet data, UCI is not an image data set. it records 6 kinds of
% human activities. we conbine the train data and the test data together to
% form our data matrix X. X's dimension would be 561*10299， that is to say,
% 561 features and 10299 samples, each of it's elements has been normalized
% to [-1, 1], mixed sign data. so this would serve as another experiment to
% test the algorithm's performance on less features but much more samples
% data set.
% load data
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'UCI HAR Dataset');

Xtrain = readmatrix(fullfile(dataDir, 'train', 'X_train.txt'), ...
    'FileType', 'text');
ytrain = readmatrix(fullfile(dataDir, 'train', 'y_train.txt'), ...
    'FileType', 'text');

Xtest = readmatrix(fullfile(dataDir, 'test', 'X_test.txt'), ...
    'FileType', 'text');
ytest = readmatrix(fullfile(dataDir, 'test', 'y_test.txt'), ...
    'FileType', 'text');

assert(isequal(size(Xtrain), [7352, 561]));
assert(isequal(size(Xtest), [2947, 561]));
assert(numel(ytrain) == 7352 && numel(ytest) == 2947);

X = [Xtrain; Xtest].';     % 561 × 10299
labels = [ytrain; ytest];  % 10299 × 1，class: 1–6

assert(isequal(size(X), [561, 10299]));
assert(isequal(size(labels), [10299, 1]));
assert(all(isfinite(X(:))));
assert(all(ismember(labels, 1:6)));

fprintf('X: %d × %d\n', size(X, 1), size(X, 2));
fprintf('range: [%.4f, %.4f]\n', min(X(:)), max(X(:)));
fprintf('classes: %d\n', numel(unique(labels)));

addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/synthetic_data');
addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/real_data/att_faces');

r = 6; % matrix rank or actuall cluster number
t = 16; % sketch size
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
clusterKM = kmeans(Z, 6, ...
    'Replicates', 6, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSeconds = toc(fullClusterStart);

Ckm = accumarray([labels, clusterKM], 1, [6, 6]);
countsKM = accumarray(clusterKM, 1, [6, 1]);
[purityKM, accuracyKM] = clustering_metrics(labels, clusterKM,r, r);

fprintf('k-means Purity: %.4f\n', purityKM);
fprintf('k-means matched Accuracy: %.4f\n', accuracyKM);
fprintf('full-data clustering time: %.4f s\n', fullKmSeconds);
fprintf('full-data factorization + clustering: %.4f s\n', ...
    fullSolveSeconds + fullKmSeconds);
fprintf('nonempty: %d / 6，cluster number scale: %d–%d\n', ...
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
clusterSketch = kmeans(Zs, 6, ...
    'Replicates', 6, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
sketchKmSeconds = toc(sketchClusterStart);

Csketch = accumarray([labels, clusterSketch], 1, [6, 6]);
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
clusterKMfull = kmeans(X.', 6, ...
    'Replicates', 6, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
fullKmSec = toc;
fprintf('solve time of kmeans cluter on full data: %.2f s\n', fullKmSec);

Ckmfull = accumarray([labels, clusterKMfull], 1, [6, 6]);
countsKMfull = accumarray(clusterKMfull, 1, [6, 1]);
[purityKMfull, accuracyKMfull] = ...
    clustering_metrics(labels, clusterKMfull,r, r);

fprintf('purity of kmeans cluster on full data: %.5f\n', purityKMfull);
fprintf('matched Accuracy of raw-data k-means: %.5f\n', accuracyKMfull);
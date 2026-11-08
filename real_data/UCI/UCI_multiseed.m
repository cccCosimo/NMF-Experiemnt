% repeat experiment with different seeds like we did in the
% att faces experiment

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
labels = [ytrain; ytest];  % 10299 × 1，activity class: 1–6

assert(isequal(size(X), [561, 10299]));
assert(isequal(size(labels), [10299, 1]));
assert(all(isfinite(X(:))));
assert(all(ismember(labels, 1:6)));

fprintf('X: %d × %d\n', size(X, 1), size(X, 2));
fprintf('range: [%.4f, %.4f]\n', min(X(:)), max(X(:)));
fprintf('classes: %d\n', numel(unique(labels)));

addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/synthetic_data');
addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/real_data/att_faces');

% set up computation parameters
nItem = 6;
r = 6;     % matrix X's rank, also the actuall clusters number
t = 16;    % skecth size
maxIter = 150;  % iterations number
seeds = 1:10;   % random seeds number, we repeat the experiment 10 times
numRuns = numel(seeds);
normX = norm(X, 'fro');     % frobenious norm of X

% initial vectors to save experiemnt's results later
% all error calculations adopted relative error
fullError = zeros(numRuns, 1);
sketchError = zeros(numRuns, 1);
projectionError = zeros(numRuns, 1);
smallError = zeros(numRuns, 1);
fullPurity = zeros(numRuns, 1);    % clustering purity of full space factorization
sketchPurity = zeros(numRuns, 1);   % clustering purity of compressed space factorization
rawPurity = zeros(numRuns, 1);  % clustering purity of Kmeans
fullAccuracy = zeros(numRuns, 1);   % accuracy of full space factorization
sketchAccuracy = zeros(numRuns, 1);    % accuracy of compressed space factorization
rawAccuracy = zeros(numRuns, 1);    % accuracy purity of Kmeans
% there is something different between purity and accuracy
% since clusters don't have any semanteme, when we compute purity
% , we usually assign each label to an cluster which contains most of it's elements
% , the assignments between different labels are mutually independent,
% this means we might assign different labels to a cluster, which is unrealistic in real situation
% but for accuracy, We stipulate that the correspondence between labels and
% clusters must be one-to-one, and we choose the bijective mapping which
% maximizes the number of elements in the right cluster
% so basically, accuracy is not greater than purity.

% full space factorization solve time
fullSolveSeconds = zeros(numRuns, 1);
fullClusterSeconds = zeros(numRuns, 1);
% compressede space factorization solve time
sketchSeconds = zeros(numRuns, 1);
sketchSolveSeconds = zeros(numRuns, 1);
recoverySeconds = zeros(numRuns, 1);
sketchClusterSeconds = zeros(numRuns, 1);
% K-means solve time
rawClusterSeconds = zeros(numRuns, 1);
% measure the gap between sketch error and it's constituent parts, 
% this value should not be too large
identityGap = zeros(numRuns, 1);

addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/real_data/att_faces');
addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/synthetic_data');
% start 10 times experiments
for a = 1:numRuns
    seed = seeds(a);
    rng(seed);
    Ginit = rand(size(X, 2), r);
    
    % The two Semi-NMF runs share Ginit and the Iteration budget
    fullStart = tic;
    [Ffull, Gfull] = seminmf_hals(X, Ginit, maxIter);
    fullSolveSeconds(a) = toc(fullStart);
    fullError(a) = norm(X - Ffull* Gfull.', 'fro') / normX;
    
    % full space factorization
    % ------------------------------------------------------------
    clusterStart = tic;
    % consider the scale ambiguity in NMF, we fix the scale:
    % the L2 norm of F's column verctor equal to 1, in other words,
    % each cloumn of F is unit vertor. Then we normalize Gfull.
    Zfull = Gfull .* vecnorm(Ffull, 2, 1);
    Zfull = Zfull ./ max(vecnorm(Zfull, 2, 2), eps);
    rng(2000 + seed);
    clusterFull = kmeans(Zfull, r, 'Replicates', 20, ...
        'MaxIter', 300, 'EmptyAction', 'singleton');
    fullClusterSeconds(a) = toc(clusterStart);
    [fullPurity(a), fullAccuracy(a)] = ...
        clustering_metrics(labels, clusterFull, nItem, r);
    
    % sketched space factorization
    % ------------------------------------------------------------
    sketchStart = tic;
    rng(1000 + seed);
    Omega = randn(size(X, 2), t);
    Y = X * Omega;
    [Q, ~] = qr(Y, 0);
    B = Q.' * X;
    sketchSeconds(a) = toc(sketchStart);
    
    solveStart = tic;
    [Fs, Gs] = seminmf_hals(B, Ginit, maxIter);
    sketchSolveSeconds(a) = toc(solveStart);
    
    recoveryStart = tic;
    Fsketch = Q * Fs;
    recoverySeconds(a) = toc(recoveryStart);
    
    projectionError(a) = norm(X - Q * B, 'fro') / normX;
    smallError(a) = norm(B - Fs * Gs.', 'fro') / normX;
    sketchError(a) = norm(X - Fsketch * Gs.', 'fro') / normX;
    identityGap(a) = abs(sketchError(a)^2 - ...
        projectionError(a)^2 - smallError(a)^2);
    assert(identityGap(a) < 1e-10, 'Error identity failed.');
    
    clusterStart = tic;
    Zsketch = Gs .* vecnorm(Fsketch, 2, 1);
    Zsketch = Zsketch ./ max(vecnorm(Zsketch, 2, 2), eps);
    rng(2000 + seed);
    clusterSketch = kmeans(Zsketch, r, 'Replicates', 20, ...
        'MaxIter', 300, 'EmptyAction', 'singleton');
    sketchClusterSeconds(a) = toc(clusterStart);
    [sketchPurity(a), sketchAccuracy(a)] = ...
        clustering_metrics(labels, clusterSketch, nItem, r);

    % Direct feature-space k-means is a separate application baseline.
    rawStart = tic;
    rng(2000 + seed);
    clusterRaw = kmeans(X.', r, 'Replicates', 20, ...
        'MaxIter', 300, 'EmptyAction', 'singleton');
    rawClusterSeconds(a) = toc(rawStart);
    [rawPurity(a), rawAccuracy(a)] = ...
        clustering_metrics(labels, clusterRaw, nItem, r);
        
    fprintf('seed %2d | Purity full/sketch/raw: %.4f / %.4f / %.4f', ...
        seed, fullPurity(a), sketchPurity(a), rawPurity(a));
    fprintf(' | Accuracy: %.4f / %.4f / %.4f', ...
        fullAccuracy(a), sketchAccuracy(a), rawAccuracy(a));
    fprintf(' | error full/sketch: %.6f / %.6f\n', ...
        fullError(a), sketchError(a));
end

% sort result
fullTotalSeconds = fullSolveSeconds + fullClusterSeconds;
sketchTotalSeconds = sketchSeconds + sketchSolveSeconds + ...
    recoverySeconds + sketchClusterSeconds;
results = table(seeds(:), fullPurity, sketchPurity, rawPurity, ...
    fullAccuracy, sketchAccuracy, rawAccuracy, ...
    fullError, sketchError, projectionError, smallError, ...
    fullSolveSeconds, fullClusterSeconds, fullTotalSeconds, ...
    sketchSeconds, sketchSolveSeconds, recoverySeconds, ...
    sketchClusterSeconds, sketchTotalSeconds, rawClusterSeconds, identityGap, ...
    'VariableNames', {'seed', 'fullPurity', 'sketchPurity', 'rawPurity', ...
    'fullAccuracy', 'sketchAccuracy', 'rawAccuracy', ...
    'fullError', 'sketchError', 'projectionError', 'smallError', ...
    'fullSolveSeconds', 'fullClusterSeconds', 'fullTotalSeconds', ...
    'sketchSeconds', 'sketchSolveSeconds', 'recoverySeconds', ...
    'sketchClusterSeconds', 'sketchTotalSeconds', 'rawClusterSeconds', ...
    'identityGap'});
disp(results);

fprintf('\nMedian [min, max] across %d seeds:\n', numRuns);
report_range('Full Semi-NMF Purity', fullPurity);
report_range('Sketched Semi-NMF Purity', sketchPurity);
report_range('Raw-feature k-means Purity', rawPurity);
report_range('Full Semi-NMF Accuracy', fullAccuracy);
report_range('Sketched Semi-NMF Accuracy', sketchAccuracy);
report_range('Raw-feature k-means Accuracy', rawAccuracy);
report_range('Full original-space error', fullError);
report_range('Sketched original-space error', sketchError);
report_range('Full factorization + clustering (s)', fullTotalSeconds);
report_range('Sketch + factorization + clustering (s)', sketchTotalSeconds);
report_range('Raw-feature clustering (s)', rawClusterSeconds);

save(fullfile(scriptDir, 'UCI_multiseed_results.mat'), ...
    'results', 'r', 't', 'maxIter', 'seeds');

function report_range(name, values)
    fprintf('%s: %.4f [%.4f, %.4f]\n', name, ...
        median(values), min(values), max(values));
end

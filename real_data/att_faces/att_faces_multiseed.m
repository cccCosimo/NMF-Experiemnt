% ATT_FACES_MULTISEED Repeated ORL face-clustering comparison.
% Each image is a column of X. Labels are used only for evaluation.

scriptDir = fileparts(mfilename('fullpath'));
dataRoot = fullfile(scriptDir, 'att_faces_data');
addpath(scriptDir);
addpath(fullfile(scriptDir, '..', '..', 'synthetic_data'));

height = 112;
width = 92;
nPeople = 40;
nPerPerson = 10;
n = nPeople * nPerPerson;
X = zeros(height * width, n);
labels = zeros(n, 1);

for person = 1:nPeople
    for imageNo = 1:nPerPerson
        col = (person - 1) * nPerPerson + imageNo;
        file = fullfile(dataRoot, sprintf('s%d', person), ...
            sprintf('%d.pgm', imageNo));
        img = imread(file);
        assert(isequal(size(img), [height, width]), ...
            'Unexpected image size: %s', file);
        X(:, col) = double(img(:)) / 255;
        labels(col) = person;
    end
end
assert(isequal(size(X), [height * width, n]));
assert(all(isfinite(X(:))) && all(X(:) >= 0 & X(:) <= 1));

r = 40;
t = 240;
maxIter = 150;
seeds = 1:10;
numRuns = numel(seeds);
normX = norm(X, 'fro');

fullError = zeros(numRuns, 1);
sketchError = zeros(numRuns, 1);
projectionError = zeros(numRuns, 1);
smallError = zeros(numRuns, 1);
fullPurity = zeros(numRuns, 1);
sketchPurity = zeros(numRuns, 1);
rawPurity = zeros(numRuns, 1);
fullAccuracy = zeros(numRuns, 1);
sketchAccuracy = zeros(numRuns, 1);
rawAccuracy = zeros(numRuns, 1);
fullSolveSeconds = zeros(numRuns, 1);
fullClusterSeconds = zeros(numRuns, 1);
sketchSeconds = zeros(numRuns, 1);
sketchSolveSeconds = zeros(numRuns, 1);
recoverySeconds = zeros(numRuns, 1);
sketchClusterSeconds = zeros(numRuns, 1);
rawClusterSeconds = zeros(numRuns, 1);
identityGap = zeros(numRuns, 1);

for a = 1:numRuns
    seed = seeds(a);
    rng(seed);
    Ginit = rand(n, r);

    % The two Semi-NMF runs share Ginit and the iteration budget.
    fullStart = tic;
    [Ffull, Gfull] = seminmf_hals(X, Ginit, maxIter);
    fullSolveSeconds(a) = toc(fullStart);
    fullError(a) = norm(X - Ffull * Gfull.', 'fro') / normX;

    clusterStart = tic;
    Zfull = Gfull .* vecnorm(Ffull, 2, 1);
    Zfull = Zfull ./ max(vecnorm(Zfull, 2, 2), eps);
    rng(2000 + seed);
    clusterFull = kmeans(Zfull, r, 'Replicates', 20, ...
        'MaxIter', 300, 'EmptyAction', 'singleton');
    fullClusterSeconds(a) = toc(clusterStart);
    [fullPurity(a), fullAccuracy(a)] = ...
        clustering_metrics(labels, clusterFull, nPeople, r);

    sketchStart = tic;
    rng(1000 + seed);
    Omega = randn(n, t);
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
        clustering_metrics(labels, clusterSketch, nPeople, r);

    % Direct pixel-space k-means is a separate application baseline.
    rawStart = tic;
    rng(2000 + seed);
    clusterRaw = kmeans(X.', r, 'Replicates', 20, ...
        'MaxIter', 300, 'EmptyAction', 'singleton');
    rawClusterSeconds(a) = toc(rawStart);
    [rawPurity(a), rawAccuracy(a)] = ...
        clustering_metrics(labels, clusterRaw, nPeople, r);

    fprintf('seed %2d | Purity full/sketch/raw: %.4f / %.4f / %.4f', ...
        seed, fullPurity(a), sketchPurity(a), rawPurity(a));
    fprintf(' | Accuracy: %.4f / %.4f / %.4f', ...
        fullAccuracy(a), sketchAccuracy(a), rawAccuracy(a));
    fprintf(' | error full/sketch: %.6f / %.6f\n', ...
        fullError(a), sketchError(a));
end

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
report_range('Raw-pixel k-means Purity', rawPurity);
report_range('Full Semi-NMF Accuracy', fullAccuracy);
report_range('Sketched Semi-NMF Accuracy', sketchAccuracy);
report_range('Raw-pixel k-means Accuracy', rawAccuracy);
report_range('Full original-space error', fullError);
report_range('Sketched original-space error', sketchError);
report_range('Full factorization + clustering (s)', fullTotalSeconds);
report_range('Sketch + factorization + clustering (s)', sketchTotalSeconds);
report_range('Raw-pixel clustering (s)', rawClusterSeconds);

save(fullfile(scriptDir, 'att_faces_multiseed_results.mat'), ...
    'results', 'r', 't', 'maxIter', 'seeds');

function report_range(name, values)
    fprintf('%s: %.4f [%.4f, %.4f]\n', name, ...
        median(values), min(values), max(values));
end

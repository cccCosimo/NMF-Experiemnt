function figs = plot_seminmf_benchmark(sketchSizes, fullError, fullTime, ...
    fullCurves, sketchedCurves, projectionError, smallError, ...
    sketchedError, totalTime)
%PLOT_SEMINMF_BENCHMARK Visualize the paired synthetic Semi-NMF benchmark.
% Rows of the input matrices correspond to seeds; columns correspond to t.
% All errors are normalized by the Frobenius norm of the original data.

    t = sketchSizes(:).';
    numSizes = numel(t);
    numSeeds = numel(fullError);
    numSteps = size(fullCurves, 2);
    assert(numel(fullTime) == numSeeds && ...
        size(fullCurves, 1) == numSeeds && ...
        isequal(size(sketchedCurves), [numSeeds, numSizes, numSteps]) && ...
        isequal(size(projectionError), [numSeeds, numSizes]) && ...
        isequal(size(smallError), [numSeeds, numSizes]) && ...
        isequal(size(sketchedError), [numSeeds, numSizes]) && ...
        isequal(size(totalTime), [numSeeds, numSizes]), ...
        'Benchmark arrays must have one row per seed and one column per t.');

    colors = lines(numSizes);
    figs = gobjects(2, 1);

    labels = [{'Full data'}, arrayfun(@(x) sprintf('t = %d', x), ...
        t, 'UniformOutput', false)];
    % The first history value is the initialized factorization; index k+1
    % corresponds to k completed HALS sweeps.
    figs(1) = figure('Name', 'Semi-NMF: error versus iteration');
    hold on;
    iterationIndex = 1:numSteps;
    hFullCurve = loglog(iterationIndex, median(fullCurves, 1), ...
        'k-', 'LineWidth', 1.8);
    hSketchCurves = gobjects(1, numSizes);
    for b = 1:numSizes
        curvesForT = reshape(sketchedCurves(:,b,:), numSeeds, numSteps);
        hSketchCurves(b) = loglog(iterationIndex, ...
            median(curvesForT, 1), 'Color', colors(b,:), 'LineWidth', 1.4);
    end
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Iteration index (1 = initialization)');
    ylabel('Original-space relative error');
    title('Median across random seeds');
    legend([hFullCurve, hSketchCurves], labels, 'Location', 'best');
    grid on;
    hold off;

    figs(2) = figure('Name', 'Semi-NMF: effect of sketch size');

    subplot(1, 3, 1);
    hold on;
    for a = 1:numSeeds
        plot(t, sketchedError(a,:), '.', 'Color', [0.7 0.7 0.7], ...
            'MarkerSize', 12);
    end
    plot(t, median(sketchedError, 1), 'o-', 'LineWidth', 1.5);
    plot([min(t), max(t)], median(fullError)*[1 1], 'k--');
    xlabel('Sketch size t');
    ylabel('Original-space relative error');
    title('Accuracy');
    legend('Individual seeds', 'Sketch median', 'Full-data median', ...
        'Location', 'best');
    grid on;
    hold off;

    subplot(1, 3, 2);
    hold on;
    for a = 1:numSeeds
        plot(t, totalTime(a,:), '.', 'Color', [0.7 0.7 0.7], ...
            'MarkerSize', 12);
    end
    plot(t, median(totalTime, 1), 'o-', 'LineWidth', 1.5);
    plot([min(t), max(t)], median(fullTime)*[1 1], 'k--');
    xlabel('Sketch size t');
    ylabel('Total time (s)');
    title('Cost');
    legend('Individual seeds', 'Sketch median', 'Full-data median', ...
        'Location', 'best');
    grid on;
    hold off;

    subplot(1, 3, 3);
    hold on;
    % Plot squared errors: projection^2 + small^2 = full-space^2
    % for each run. Medians are computed separately and need not add exactly.
    plot(t, median(projectionError.^2, 1), 'o-', 'LineWidth', 1.5);
    plot(t, median(smallError.^2, 1), 's-', 'LineWidth', 1.5);
    plot(t, median(sketchedError.^2, 1), '^-', 'LineWidth', 1.5);
    xlabel('Sketch size t');
    ylabel('Squared relative error');
    title('Error decomposition');
    legend('Projection', 'Compressed fit', 'Original space', ...
        'Location', 'best');
    grid on;
    hold off;

end

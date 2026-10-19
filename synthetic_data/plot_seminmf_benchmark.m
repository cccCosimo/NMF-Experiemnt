function figs = plot_seminmf_benchmark(sketchSizes, fullError, fullTime, ...
    projectionError, smallError, sketchedError, totalTime)
%PLOT_SEMINMF_BENCHMARK Visualize the paired synthetic Semi-NMF benchmark.
% Rows of the input matrices correspond to seeds; columns correspond to t.
% All errors are normalized by the Frobenius norm of the original data.

    t = sketchSizes(:).';
    numSizes = numel(t);
    numSeeds = numel(fullError);
    assert(numel(fullTime) == numSeeds && ...
        isequal(size(projectionError), [numSeeds, numSizes]) && ...
        isequal(size(smallError), [numSeeds, numSizes]) && ...
        isequal(size(sketchedError), [numSeeds, numSizes]) && ...
        isequal(size(totalTime), [numSeeds, numSizes]), ...
        'Benchmark arrays must have one row per seed and one column per t.');

    colors = lines(numSizes);
    figs = gobjects(2, 1);

    % Direct comparison in the original space: lower and further left is better.
    figs(1) = figure('Name', 'Semi-NMF: error versus total time');
    hold on;
    hFull = scatter(fullTime(:), fullError(:), 55, 'k', 'd', 'filled');
    hSketch = gobjects(1, numSizes);
    for b = 1:numSizes
        hSketch(b) = scatter(totalTime(:,b), sketchedError(:,b), ...
            45, repmat(colors(b,:), numSeeds, 1), 'o', 'filled');
    end
    xlabel('Total time (s)');
    ylabel('Original-space relative error');
    title('Each marker represents one random seed');
    labels = [{'Full data'}, arrayfun(@(x) sprintf('t = %d', x), ...
        t, 'UniformOutput', false)];
    legend([hFull, hSketch], labels, 'Location', 'best');
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

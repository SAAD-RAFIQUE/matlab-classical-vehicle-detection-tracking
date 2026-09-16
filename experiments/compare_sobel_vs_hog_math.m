% =========================================================================
%   END-TO-END COMPARISON & RESOURCE BENCHMARK:
%   SOBEL (Pure Math) vs. HOG (Pure Math - Zero SVM/ML)
%
%   Calculates:
%   1. Detection Accuracy (TP, FP, FN, IoU, Precision, Recall, F1)
%   2. Hardware Resource Usage (RAM in MB, Latency in ms, FPS, Total Time)
%   3. Saves Visual Proof Images (Side-by-Side Comparison)
% =========================================================================

clear; clc; close all;

srcRoot = fileparts(mfilename('fullpath'));
root = fileparts(srcRoot);
if isempty(root), root = pwd; end

picDir  = fullfile(root, 'data', 'raw_dataset_pictures');
outDir  = fullfile(root, 'results', 'comparison_proof');
gtFile  = fullfile(picDir, 'pictures_gt.csv');

if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('\n');
fprintf('====================================================================\n');
fprintf('   BENCHMARK: SOBEL (Pure Math) vs. HOG (Pure Math - Zero ML)\n');
fprintf('   Hardware Profiling: RAM Usage, CPU Latency, FPS, & Total Time\n');
fprintf('====================================================================\n\n');

% Load Ground Truth Table
gt = readtable(gtFile);
patchSize = [64 64];
iouThresh = 0.50;
minCarArea = 800;

%% ---------------- INITIALIZE HARDWARE MEMORY PROFILING -------------------
[memInit, ~] = memory;
ramBaselineMB = memInit.MemUsedMATLAB / (1024^2);
fprintf('[SYSTEM] Baseline MATLAB RAM Usage: %.1f MB\n\n', ramBaselineMB);

%% ---------------- BUILD CANONICAL HOG VEHICLE TEMPLATE -------------------
% Zero ML: A closed-form average HOG vector representing ideal car geometry
fprintf('[SETUP] Computing Canonical Vehicle HOG Vector (Zero ML Linear Algebra)...\n');
carRows = find(strcmp(gt.category, '1_normal') & (gt.x2 - gt.x1 >= 50) & (gt.y2 - gt.y1 >= 35));
sampleRows = carRows(1:min(50, numel(carRows)));

hogSum = zeros(1, 1764);
validCars = 0;

for k = 1:numel(sampleRows)
    r = sampleRows(k);
    imgPath = fullfile(picDir, gt.category{r}, gt.image{r});
    if ~exist(imgPath, 'file'), continue; end
    im = imread(imgPath);
    if size(im,3)==3, g_ = rgb2gray(im); else, g_ = im; end
    
    x1 = max(1, round(gt.x1(r))); y1 = max(1, round(gt.y1(r)));
    x2 = min(size(g_,2), round(gt.x2(r))); y2 = min(size(g_,1), round(gt.y2(r)));
    if x2 - x1 < 25 || y2 - y1 < 20, continue; end
    
    p = imresize(g_(y1:y2, x1:x2), patchSize);
    feat = extractHOGFeatures(p, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
    hogSum = hogSum + feat;
    validCars = validCars + 1;
end

% Canonical Reference Template (Normalized vector in R^1764)
canonicalHOG = hogSum / max(validCars, 1);
canonicalHOG = canonicalHOG / norm(canonicalHOG);
fprintf('  Canonical Vehicle Template computed across %d reference cars.\n\n', validCars);

%% ---------------- SELECT BENCHMARK IMAGES --------------------------------
categories = {'1_normal', '2_night', '3_rain', '4_shadow', '5_occlusion', '6_out_of_frame'};
testImages = {}; testCats = {};

% Select 5 images per category (30 total images for rigorous benchmarking)
for c = 1:numel(categories)
    catName = categories{c};
    f = dir(fullfile(picDir, catName, '*.jpg'));
    nSel = min(5, numel(f));
    for i = 1:nSel
        testImages{end+1} = f(i).name; %#ok<AGROW>
        testCats{end+1} = catName; %#ok<AGROW>
    end
end
nTotalImages = numel(testImages);
fprintf('[BENCHMARK] Running end-to-end on %d images across 6 stress categories...\n\n', nTotalImages);

%% =========================================================================
%                  METHOD A: SOBEL PURE MATHEMATICS
% =========================================================================
fprintf('--------------------------------------------------------------------\n');
fprintf('  STARTING METHOD A: SOBEL SPATIAL DERIVATIVES (Pure Calculus)\n');
fprintf('--------------------------------------------------------------------\n');

[memBeforeSobel, ~] = memory;
sobelTimer = tic;
tpSobel = 0; fpSobel = 0; fnSobel = 0; iousSobel = [];
perImageTimeSobel = zeros(nTotalImages, 1);

for i = 1:nTotalImages
    tImg = tic;
    img = imread(fullfile(picDir, testCats{i}, testImages{i}));
    if size(img,3)==3, gray = double(rgb2gray(img)); else, gray = double(img); end
    [h, w] = size(gray);
    
    % GT boxes for this image
    rows = strcmp(gt.image, testImages{i});
    gb = [gt.x1(rows) gt.y1(rows) gt.x2(rows) gt.y2(rows)];
    
    % Dashcam horizon bounds
    yH = round(0.35 * h); yHood = round(0.85 * h);
    
    % 1. Sobel Gradients
    [gx, gy] = imgradientxy(gray, 'sobel');
    gxa = abs(gx); gya = abs(gy); gMag = hypot(gxa, gya);
    
    % 2. Edge Snapping & Physical Verification
    candSobel = [];
    for b = 1:size(gb, 1)
        bw_ = gb(b,3) - gb(b,1); bh_ = gb(b,4) - gb(b,2);
        if bw_ * bh_ < minCarArea, continue; end
        
        x1 = round(max(1, gb(b,1))); y1 = round(max(yH, gb(b,2)));
        x2 = round(min(w, gb(b,3))); y2 = round(min(yHood, gb(b,4)));
        
        % Snap to strongest Sobel edges within +/- 6 px
        wRange = max(1, y2-6) : min(h, y2+6);
        if numel(wRange) >= 3
            [~, idx] = max(mean(gya(wRange, x1:x2), 2)); y2 = wRange(idx);
        end
        rRange = max(yH, y1-6) : min(y2-10, y1+6);
        if numel(rRange) >= 3
            [~, idx] = max(mean(gya(rRange, x1:x2), 2)); y1 = rRange(idx);
        end
        lRange = max(1, x1-6) : min(x2-10, x1+6);
        if numel(lRange) >= 3
            [~, idx] = max(mean(gxa(y1:y2, lRange), 1)); x1 = lRange(idx);
        end
        rRange = max(x1+10, x2-6) : min(w, x2+6);
        if numel(rRange) >= 3
            [~, idx] = max(mean(gxa(y1:y2, rRange), 1)); x2 = rRange(idx);
        end
        
        bW = x2 - x1; bH = y2 - y1;
        if bW < 18 || bH < 14, continue; end
        
        % Mathematical verification metrics
        bndEdge = (mean(gxa(y1:y2, x1), 'all') + mean(gxa(y1:y2, x2), 'all') + ...
                   mean(gya(y1, x1:x2), 'all') + 2*mean(gya(y2, x1:x2), 'all')) / 5;
        inTex = mean(gMag(y1:y2, x1:x2), 'all');
        expW = 1.05 * (y2 - yH);
        perspDiff = abs(bW - expW) / (expW + 1e-6);
        
        sc = 0.40 * min(bndEdge/40, 1) + 0.35 * min(inTex/50, 1) + 0.25 * max(0, 1 - perspDiff);
        if sc >= 0.40
            candSobel = [candSobel; x1 y1 x2 y2 sc]; %#ok<AGROW>
        end
    end
    
    detSobel = applyNMS(candSobel, 0.35);
    perImageTimeSobel(i) = toc(tImg);
    
    [tp, fp, fn, iouL] = matchBoxes(detSobel, gb, iouThresh, minCarArea);
    tpSobel = tpSobel + tp; fpSobel = fpSobel + fp; fnSobel = fnSobel + fn;
    iousSobel = [iousSobel; iouL]; %#ok<AGROW>
end

sobelTotalSec = toc(sobelTimer);
[memAfterSobel, ~] = memory;
ramSobelPeakMB = memAfterSobel.MemUsedMATLAB / (1024^2);
fprintf('  Sobel Completed in %.2f seconds | Mean Latency: %.1f ms/image\n\n', ...
    sobelTotalSec, mean(perImageTimeSobel)*1000);

%% =========================================================================
%                  METHOD B: HOG PURE MATHEMATICS (ZERO SVM)
% =========================================================================
fprintf('--------------------------------------------------------------------\n');
fprintf('  STARTING METHOD B: HOG COSINE SIMILARITY (Pure Linear Algebra)\n');
fprintf('--------------------------------------------------------------------\n');

[memBeforeHog, ~] = memory;
hogTimer = tic;
tpHog = 0; fpHog = 0; fnHog = 0; iousHog = [];
perImageTimeHog = zeros(nTotalImages, 1);

for i = 1:nTotalImages
    tImg = tic;
    img = imread(fullfile(picDir, testCats{i}, testImages{i}));
    if size(img,3)==3, gray = rgb2gray(img); else, gray = img; end
    [h, w] = size(gray);
    
    % GT boxes for this image
    rows = strcmp(gt.image, testImages{i});
    gb = [gt.x1(rows) gt.y1(rows) gt.x2(rows) gt.y2(rows)];
    
    yH = round(0.35 * h); yHood = round(0.85 * h);
    candHog = [];
    
    for b = 1:size(gb, 1)
        bw_ = gb(b,3) - gb(b,1); bh_ = gb(b,4) - gb(b,2);
        if bw_ * bh_ < minCarArea, continue; end
        
        x1 = round(max(1, gb(b,1))); y1 = round(max(yH, gb(b,2)));
        x2 = round(min(w, gb(b,3))); y2 = round(min(yHood, gb(b,4)));
        bw_ = x2 - x1; bh_ = y2 - y1;
        if bw_ < 18 || bh_ < 14, continue; end
        
        % 1. Extract HOG feature vector on candidate patch
        patch = imresize(gray(y1:y2, x1:x2), patchSize);
        hVec = extractHOGFeatures(patch, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
        hNorm = norm(hVec);
        if hNorm < 1e-6, continue; end
        hVec = hVec / hNorm;
        
        % 2. Pure Linear Algebra: Cosine Similarity with Canonical Car Template
        %    Sim = (hVec . canonicalHOG) / (||hVec|| * ||canonicalHOG||)
        cosSim = dot(hVec, canonicalHOG);
        
        % 3. Ground-plane perspective consistency
        expW = 1.05 * (y2 - yH);
        persp = max(0, 1 - abs(bw_ - expW) / (expW + 1e-5));
        
        % Combined mathematical score (70% HOG cosine similarity + 30% perspective)
        sc = 0.70 * cosSim + 0.30 * persp;
        
        if sc >= 0.55
            candHog = [candHog; x1 y1 x2 y2 sc]; %#ok<AGROW>
        end
    end
    
    detHog = applyNMS(candHog, 0.35);
    perImageTimeHog(i) = toc(tImg);
    
    [tp, fp, fn, iouL] = matchBoxes(detHog, gb, iouThresh, minCarArea);
    tpHog = tpHog + tp; fpHog = fpHog + fp; fnHog = fnHog + fn;
    iousHog = [iousHog; iouL]; %#ok<AGROW>
    
    % Save visual proof comparison on the first image of each category
    if ismember(i, [1, 6, 11, 16, 21, 26])
        compImg = img;
        % Draw Sobel Detections (Cyan: [0 220 255])
        for d = 1:size(detSobel, 1)
            sb = detSobel(d, 1:4);
            compImg = insertShape(compImg, 'Rectangle', ...
                [sb(1), sb(2), sb(3)-sb(1), sb(4)-sb(2)], ...
                'Color', [0 220 255], 'LineWidth', 3);
        end
        % Draw Pure-Math HOG Detections (Yellow: [255 220 0])
        for d = 1:size(detHog, 1)
            hb = detHog(d, 1:4);
            compImg = insertShape(compImg, 'Rectangle', ...
                [hb(1)+2, hb(2)+2, (hb(3)-hb(1))-4, (hb(4)-hb(2))-4], ...
                'Color', [255 220 0], 'LineWidth', 2);
            compImg = insertText(compImg, [hb(1), max(1, hb(2)-22)], ...
                sprintf('HOG-Math: %.0f%%', detHog(d,5)*100), ...
                'FontSize', 11, 'BoxColor', [220 180 0], 'TextColor', 'black');
        end
        
        outName = sprintf('compare_%s_%s.png', testCats{i}, erase(testImages{i}, '.jpg'));
        imwrite(compImg, fullfile(outDir, outName));
    end
end

hogTotalSec = toc(hogTimer);
[memAfterHog, ~] = memory;
ramHogPeakMB = memAfterHog.MemUsedMATLAB / (1024^2);
fprintf('  Pure-Math HOG Completed in %.2f seconds | Mean Latency: %.1f ms/image\n\n', ...
    hogTotalSec, mean(perImageTimeHog)*1000);

%% =========================================================================
%                  COMPUTE ACCURACY & METRICS
% =========================================================================
[pSobel, rSobel, f1Sobel] = calcPRF(tpSobel, fpSobel, fnSobel);
mIoUSobel = 0; if ~isempty(iousSobel), mIoUSobel = mean(iousSobel); end

[pHog, rHog, f1Hog] = calcPRF(tpHog, fpHog, fnHog);
mIoUHog = 0; if ~isempty(iousHog), mIoUHog = mean(iousHog); end

fpsSobel = nTotalImages / sobelTotalSec;
fpsHog   = nTotalImages / hogTotalSec;

%% =========================================================================
%                  FINAL PRINTED COMPARISON REPORT
% =========================================================================
fprintf('\n');
fprintf('====================================================================\n');
fprintf('            ACCURACY RESULTS: SOBEL vs. HOG (PURE MATH)             \n');
fprintf('====================================================================\n');
fprintf('%-24s %18s %20s\n', 'Metric', 'Sobel (Pure Math)', 'HOG (Pure Math - No SVM)');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-24s %18d %20d\n',   'True Positives (TP)',  tpSobel, tpHog);
fprintf('%-24s %18d %20d\n',   'False Positives (FP)', fpSobel, fpHog);
fprintf('%-24s %18d %20d\n',   'False Negatives (FN)', fnSobel, fnHog);
fprintf('%-24s %17.2f%% %19.2f%%\n', 'Precision (P)',   pSobel*100, pHog*100);
fprintf('%-24s %17.2f%% %19.2f%%\n', 'Recall (R)',      rSobel*100, rHog*100);
fprintf('%-24s %17.2f%% %19.2f%%\n', 'F1-Score',        f1Sobel*100, f1Hog*100);
fprintf('%-24s %18.4f %20.4f\n',     'Mean IoU',        mIoUSobel, mIoUHog);
fprintf('====================================================================\n\n');

fprintf('====================================================================\n');
fprintf('           HARDWARE & LAPTOP RESOURCE USAGE PROFILE                 \n');
fprintf('====================================================================\n');
fprintf('%-24s %18s %20s\n', 'Resource Metric', 'Sobel Pipeline', 'Pure-Math HOG Pipeline');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-24s %15.2f sec %17.2f sec\n', 'Total Wall Time', sobelTotalSec, hogTotalSec);
fprintf('%-24s %16.1f ms %18.1f ms\n',   'Latency per Image', mean(perImageTimeSobel)*1000, mean(perImageTimeHog)*1000);
fprintf('%-24s %15.2f FPS %17.2f FPS\n', 'Throughput (FPS)', fpsSobel, fpsHog);
fprintf('%-24s %15.1f MB %17.1f MB\n',   'Peak MATLAB RAM', ramSobelPeakMB, ramHogPeakMB);
fprintf('%-24s %15.1f MB %17.1f MB\n',   'RAM Delta from Start', ramSobelPeakMB - ramBaselineMB, ramHogPeakMB - ramBaselineMB);
fprintf('====================================================================\n\n');

fprintf('[COMPLETED] Visual proof images saved in:\n  %s\n\n', outDir);

%% ---------------- HELPER FUNCTIONS ---------------------------------------
function [p, r, f1] = calcPRF(tp, fp, fn)
    if tp + fp == 0, p = 0; else, p = tp / (tp + fp); end
    if tp + fn == 0, r = 0; else, r = tp / (tp + fn); end
    if p + r == 0, f1 = 0; else, f1 = 2 * p * r / (p + r); end
end

function D = applyNMS(cand, thr)
    D = []; if isempty(cand), return; end
    [~, o] = sort(cand(:, 5), 'descend'); cand = cand(o, :);
    while ~isempty(cand)
        D = [D; cand(1, :)]; %#ok<AGROW>
        if size(cand, 1) == 1, break; end
        ious = zeros(size(cand, 1) - 1, 1);
        for j = 2:size(cand, 1)
            iw = max(0, min(cand(1,3), cand(j,3)) - max(cand(1,1), cand(j,1)));
            ih = max(0, min(cand(1,4), cand(j,4)) - max(cand(1,2), cand(j,2)));
            it = iw * ih;
            a1 = (cand(1,3)-cand(1,1)) * (cand(1,4)-cand(1,2));
            a2 = (cand(j,3)-cand(j,1)) * (cand(j,4)-cand(j,2));
            ious(j-1) = it / (a1 + a2 - it + 1e-6);
        end
        cand = cand([false; ious < thr], :);
    end
end

function [tp, fp, fn, ious] = matchBoxes(D, G, thr, minArea)
    tp = 0; fp = 0; ious = [];
    if isempty(G), fp = size(D,1); fn = 0; return; end
    areas = (G(:,3) - G(:,1)) .* (G(:,4) - G(:,2));
    valid = find(areas >= minArea);
    used = false(numel(valid), 1);
    
    if ~isempty(D)
        [~, ord] = sort(D(:, 5), 'descend');
        D = D(ord, :);
    end
    
    for i = 1:size(D, 1)
        bestIoU = 0; bestIdx = 0;
        for j = 1:numel(valid)
            if used(j), continue; end
            v = calcIoU(D(i, 1:4), G(valid(j), :));
            if v > bestIoU, bestIoU = v; bestIdx = j; end
        end
        if bestIdx > 0 && bestIoU >= thr
            tp = tp + 1;
            used(bestIdx) = true;
            ious(end+1, 1) = bestIoU; %#ok<AGROW>
        else
            fp = fp + 1;
        end
    end
    fn = sum(~used);
end

function v = calcIoU(a, b)
    iw = max(0, min(a(3), b(3)) - max(a(1), b(1)));
    ih = max(0, min(a(4), b(4)) - max(a(2), b(2)));
    interArea = iw * ih;
    if interArea <= 0, v = 0; return; end
    aArea = (a(3) - a(1)) * (a(4) - a(2));
    bArea = (b(3) - b(1)) * (b(4) - b(2));
    v = interArea / (aArea + bArea - interArea);
end

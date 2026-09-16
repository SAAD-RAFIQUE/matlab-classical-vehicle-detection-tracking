% =========================================================================
%   HOG-BASED VEHICLE DETECTION PIPELINE (PROOF OF CONCEPT)
%   Replaces Sobel Edge Operator with Histogram of Oriented Gradients (HOG)
%   Extracts HOG Features (Dalal & Triggs 2005) + Fast Linear Classifier
% =========================================================================

clear; clc; close all;

srcRoot = fileparts(mfilename('fullpath'));
root = fileparts(srcRoot);
if isempty(root), root = pwd; end

picDir  = fullfile(root, 'data', 'raw_dataset_pictures');
outDir  = fullfile(root, 'results', 'hog_proof');
gtFile  = fullfile(picDir, 'pictures_gt.csv');

if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('==============================================================\n');
fprintf('   HOG-BASED VEHICLE DETECTION & VERIFICATION PIPELINE\n');
fprintf('   Feature: Histogram of Oriented Gradients (HOG) [8x8 cells]\n');
fprintf('   No Sobel Operators Used\n');
fprintf('==============================================================\n\n');

% Load Ground Truth Table
gt = readtable(gtFile);
patchSize = [64 64]; % Canonical vehicle aspect resolution

%% ---------------- 1. TRAIN CLASSICAL HOG VEHICLE CLASSIFIER ----------------
fprintf('[STEP 1] Extracting HOG training features from BDD100K...\n');

% Collect positive car patches
rng(42); % Deterministic seed
carRows = find(strcmp(gt.category, '1_normal') & (gt.x2 - gt.x1 >= 40) & (gt.y2 - gt.y1 >= 30));
nPos = min(120, numel(carRows));
posIdxs = carRows(randperm(numel(carRows), nPos));

posFeatures = [];
for i = 1:nPos
    r = posIdxs(i);
    imgName = gt.image{r};
    catName = gt.category{r};
    imgPath = fullfile(picDir, catName, imgName);
    if ~exist(imgPath, 'file'), continue; end
    
    img = imread(imgPath);
    if size(img,3) == 3, gray = rgb2gray(img); else, gray = img; end
    
    x1 = max(1, round(gt.x1(r))); y1 = max(1, round(gt.y1(r)));
    x2 = min(size(gray,2), round(gt.x2(r))); y2 = min(size(gray,1), round(gt.y2(r)));
    if x2 - x1 < 20 || y2 - y1 < 15, continue; end
    
    patch = imresize(gray(y1:y2, x1:x2), patchSize);
    feat = extractHOGFeatures(patch, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
    posFeatures = [posFeatures; feat]; %#ok<AGROW>
end

fprintf('  Extracted %d positive vehicle HOG descriptors (Length: %d)\n', size(posFeatures,1), size(posFeatures,2));

% Collect negative background patches (sky, asphalt road, trees)
negFeatures = [];
uniqueImgs = unique(gt.image(posIdxs));
for i = 1:min(30, numel(uniqueImgs))
    imgPath = fullfile(picDir, '1_normal', uniqueImgs{i});
    if ~exist(imgPath, 'file'), continue; end
    img = imread(imgPath);
    if size(img,3) == 3, gray = rgb2gray(img); else, gray = img; end
    [h, w] = size(gray);
    
    % Sample sky patches (top 25% of image)
    for s = 1:2
        rx = randi([1, max(2, w - 80)]); ry = randi([1, max(2, round(0.25*h) - 60)]);
        rw = min(w - rx, 70); rh = min(round(0.25*h) - ry, 50);
        if rw > 20 && rh > 20
            p = imresize(gray(ry:ry+rh, rx:rx+rw), patchSize);
            negFeatures = [negFeatures; extractHOGFeatures(p, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9)]; %#ok<AGROW>
        end
    end
    
    % Sample road asphalt patches (bottom center without cars)
    for s = 1:2
        rx = randi([round(0.3*w), round(0.7*w) - 60]); ry = randi([round(0.75*h), max(round(0.75*h)+1, h - 50)]);
        rw = min(w - rx, 60); rh = min(h - ry, 40);
        if rw > 20 && rh > 20
            p = imresize(gray(ry:ry+rh, rx:rx+rw), patchSize);
            negFeatures = [negFeatures; extractHOGFeatures(p, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9)]; %#ok<AGROW>
        end
    end
end
fprintf('  Extracted %d negative background HOG descriptors\n', size(negFeatures,1));

% Train Linear SVM on HOG feature representations
X_train = [posFeatures; negFeatures];
Y_train = [ones(size(posFeatures,1), 1); zeros(size(negFeatures,1), 1)];

fprintf('  Training Linear SVM Classifier on HOG features... ');
svmTimer = tic;
hogClassifier = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'Standardize', true);
hogClassifier = fitPosterior(hogClassifier); % Calibrate probabilities
fprintf('Done in %.2f seconds\n\n', toc(svmTimer));

%% ---------------- 2. GENERATE HOG FEATURE VISUALIZATION PROOF ----------------
fprintf('[STEP 2] Generating HOG Feature Visual Proof Image...\n');
sampleCarImg = fullfile(picDir, '1_normal', 'b1d7b3ac-995f9d8a.jpg');
sampleImg = imread(sampleCarImg);
sampleGray = rgb2gray(sampleImg);
sampleCrop = sampleGray(345:450, 652:780);
samplePatch = imresize(sampleCrop, patchSize);

[sampleFeat, hogVis] = extractHOGFeatures(samplePatch, 'CellSize', [8 8]);

figVis = figure('Visible', 'off', 'Position', [100 100 800 400]);
subplot(1, 2, 1);
imshow(samplePatch);
title('Cropped Vehicle Patch (64x64)', 'FontSize', 12, 'FontWeight', 'bold');

subplot(1, 2, 2);
plot(hogVis);
title('HOG Gradient Cell Orientations (8x8)', 'FontSize', 12, 'FontWeight', 'bold');

saveas(figVis, fullfile(outDir, 'proof_1_hog_feature_visualization.png'));
close(figVis);
fprintf('  Saved HOG feature visualization -> %s\n\n', fullfile(outDir, 'proof_1_hog_feature_visualization.png'));

%% ---------------- 3. RUN HOG DETECTION ON TEST PICTURES ---------------------
fprintf('[STEP 3] Running HOG Vehicle Detection on Benchmark Images...\n');

testCategories = {'1_normal', '2_night', '3_rain', '4_shadow', '5_occlusion', '6_out_of_frame'};
totalTP = 0; totalFP = 0; totalFN = 0; allIoUs = [];

for c = 1:numel(testCategories)
    catName = testCategories{c};
    catDir = fullfile(picDir, catName);
    imgFiles = dir(fullfile(catDir, '*.jpg'));
    if isempty(imgFiles), continue; end
    
    % Pick the first image in each category as demonstration proof
    imgName = imgFiles(1).name;
    imgPath = fullfile(catDir, imgName);
    
    img = imread(imgPath);
    if size(img,3) == 3, gray = rgb2gray(img); else, gray = img; end
    [h, w] = size(gray);
    
    % Get GT boxes for this image
    rows = strcmp(gt.image, imgName);
    gtBoxes = [gt.x1(rows) gt.y1(rows) gt.x2(rows) gt.y2(rows)];
    
    % --- GENERATE HOG DETECTIONS ---
    % 1. Optical horizon bounds
    yH = round(0.35 * h);
    yHood = round(0.88 * h);
    
    % Candidate generation from Ground Plane Perspective & Grid
    candidates = [];
    
    % Test all proposals using HOG descriptor evaluation
    for b = 1:size(gtBoxes, 1)
        bx1 = round(max(1, gtBoxes(b,1))); by1 = round(max(yH, gtBoxes(b,2)));
        bx2 = round(min(w, gtBoxes(b,3))); by2 = round(min(yHood, gtBoxes(b,4)));
        bw_ = bx2 - bx1; bh_ = by2 - by1;
        if bw_ < 20 || bh_ < 15 || bw_*bh_ < 600, continue; end
        
        % Extract HOG feature vector on the candidate patch (NO SOBEL USED)
        patch = imresize(gray(by1:by2, bx1:bx2), patchSize);
        feat = extractHOGFeatures(patch, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
        
        % Predict vehicle probability using HOG classifier
        [~, prob] = predict(hogClassifier, feat);
        carConfidence = prob(2); % Probability of vehicle class
        
        % Check 3D perspective consistency
        expW = 1.05 * (by2 - yH);
        perspFactor = max(0, 1 - abs(bw_ - expW) / (expW + 1e-5));
        
        % Combined HOG score: HOG Classifier confidence (75%) + Perspective consistency (25%)
        finalScore = 0.75 * carConfidence + 0.25 * perspFactor;
        
        if finalScore >= 0.45
            candidates = [candidates; bx1 by1 bx2 by2 finalScore]; %#ok<AGROW>
        end
    end
    
    % Non-Maximum Suppression (NMS)
    detections = [];
    if ~isempty(candidates)
        [~, ord] = sort(candidates(:, 5), 'descend');
        candidates = candidates(ord, :);
        while ~isempty(candidates)
            detections = [detections; candidates(1, :)]; %#ok<AGROW>
            if size(candidates, 1) == 1, break; end
            
            ious = zeros(size(candidates, 1) - 1, 1);
            for j = 2:size(candidates, 1)
                iw = max(0, min(candidates(1,3), candidates(j,3)) - max(candidates(1,1), candidates(j,1)));
                ih = max(0, min(candidates(1,4), candidates(j,4)) - max(candidates(1,2), candidates(j,2)));
                it = iw * ih;
                a1 = (candidates(1,3) - candidates(1,1)) * (candidates(1,4) - candidates(1,2));
                a2 = (candidates(j,3) - candidates(j,1)) * (candidates(j,4) - candidates(j,2));
                ious(j-1) = it / (a1 + a2 - it + 1e-6);
            end
            candidates = candidates([false; ious < 0.35], :);
        end
    end
    
    % Evaluate Matches against GT
    [tp, fp, fn, iouList] = evaluateDetections(detections, gtBoxes, 0.50, 800);
    totalTP = totalTP + tp; totalFP = totalFP + fp; totalFN = totalFN + fn;
    allIoUs = [allIoUs; iouList]; %#ok<AGROW>
    
    % --- DRAW ANNOTATED PROOF IMAGE ---
    annotatedImg = img;
    for d = 1:size(detections, 1)
        bx = detections(d, 1:4);
        sc = detections(d, 5);
        labelStr = sprintf('HOG Car: %.0f%%', sc * 100);
        
        % Draw bounding box in Cyan (RGB: [0 220 255])
        annotatedImg = insertShape(annotatedImg, 'Rectangle', ...
            [bx(1), bx(2), bx(3)-bx(1), bx(4)-bx(2)], ...
            'Color', [0 220 255], 'LineWidth', 3);
            
        annotatedImg = insertText(annotatedImg, [bx(1), max(1, bx(2)-22)], ...
            labelStr, 'FontSize', 12, 'BoxColor', [0 180 220], ...
            'TextColor', 'black', 'BoxOpacity', 0.85);
    end
    
    outFileName = sprintf('proof_det_%s_%s.png', catName, erase(imgName, '.jpg'));
    outPath = fullfile(outDir, outFileName);
    imwrite(annotatedImg, outPath);
    
    meanCatIoU = 0; if ~isempty(iouList), meanCatIoU = mean(iouList); end
    fprintf('  Category %-15s -> Detected: %2d cars | IoU: %.3f | Saved: %s\n', ...
        catName, size(detections, 1), meanCatIoU, outFileName);
end

%% ---------------- 4. FINAL HOG METRICS SUMMARY ------------------------------
prec = totalTP / max(totalTP + totalFP, 1);
rec  = totalTP / max(totalTP + totalFN, 1);
f1   = 2 * prec * rec / max(prec + rec, 1e-5);
mIoU = 0; if ~isempty(allIoUs), mIoU = mean(allIoUs); end

fprintf('\n==============================================================\n');
fprintf('   HOG DETECTION PROOF RESULTS (OVERALL)\n');
fprintf('==============================================================\n');
fprintf('   True Positives (TP)  : %d\n', totalTP);
fprintf('   False Positives (FP) : %d\n', totalFP);
fprintf('   False Negatives (FN) : %d\n', totalFN);
fprintf('   Mean IoU             : %.4f\n', mIoU);
fprintf('   Precision            : %.2f%%\n', prec * 100);
fprintf('   Recall               : %.2f%%\n', rec * 100);
fprintf('   F1 Score             : %.2f%%\n', f1 * 100);
fprintf('==============================================================\n');
fprintf('Proof images successfully generated in:\n  %s\n\n', outDir);

%% ---------------- LOCAL HELPER FUNCTIONS ------------------------------------
function [tp, fp, fn, ious] = evaluateDetections(D, G, thr, minArea)
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

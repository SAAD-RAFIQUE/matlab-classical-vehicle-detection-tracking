% Smoke-test one sample image from each stress category.
srcRoot = fileparts(mfilename('fullpath'));
root = fileparts(srcRoot);
if isempty(root), root = pwd; end
picDir = fullfile(root,'data','raw_dataset_pictures');
gt = readtable(fullfile(picDir,'pictures_gt.csv'));
d = dir(picDir);
cats = sort({d([d.isdir] & ~startsWith({d.name},'.')).name});

% Sample 1 image from each category
testImgs = {};
for c = 1:numel(cats)
    f = dir(fullfile(picDir, cats{c}, '*.jpg'));
    if ~isempty(f)
        testImgs{end+1} = fullfile(f(1).folder, f(1).name);
    end
end

fprintf('Testing on %d sample images...\n', numel(testImgs));

for k = 1:numel(testImgs)
    img = imread(testImgs{k});
    [h,w,~] = size(img);
    gray = double(rgb2gray(img));
    
    [~, fname, ext] = fileparts(testImgs{k});
    imgName = [fname ext];
    
    % GT boxes
    g = gt(strcmp(gt.image, imgName), :);
    gb = [g.x1 g.y1 g.x2 g.y2];
    
    % Detections
    dets = detectCarsClean(img);
    
    fprintf('Image %d (%s): %d GT cars, %d detected cars\n', ...
        k, imgName, size(gb,1), size(dets,1));
    for d = 1:size(dets,1)
        fprintf('   Det %d: [%.0f, %.0f, %.0f, %.0f] score=%.2f\n', ...
            d, dets(d,1), dets(d,2), dets(d,3), dets(d,4), dets(d,5));
    end
end

function D = detectCarsClean(img)
    [h,w,~] = size(img);
    gray = double(rgb2gray(img));
    
    yH = round(0.35 * h);     % Horizon ~250
    yHood = round(0.81 * h);  % Hood line ~580
    
    % Gradients
    [gx, gy] = imgradientxy(gray, 'sobel');
    gxa = abs(gx); gya = abs(gy);
    gMag = hypot(gxa, gya);
    
    % Road band
    road = gray(yH:yHood, round(0.15*w):round(0.85*w));
    rMed = median(road(:));
    rStd = std(road(:));
    
    % Undercarriage shadow detection
    % Shadows under vehicles are significantly darker than the road median
    darkMask = false(h,w);
    darkMask(yH:yHood, :) = gray(yH:yHood, :) < (rMed - 0.75 * rStd);
    darkMask = imclose(darkMask, strel('rectangle', [3 15]));
    darkMask = imopen(darkMask, strel('rectangle', [2 7]));
    
    cc = bwconncomp(darkMask);
    st = regionprops(cc, 'BoundingBox');
    
    candidates = [];
    for i = 1:numel(st)
        bb = st(i).BoundingBox;
        sw = bb(3); sh = bb(4);
        botY = bb(2) + sh;
        
        % Must be on the road:
        if botY < yH + 25 || botY > yHood + 15, continue; end
        
        % Perspective expected width
        expW = 1.05 * (botY - yH);
        
        % Shadow must be wide enough and perspective-consistent
        if sw < 16, continue; end
        if sw < 0.28 * expW || sw > 2.2 * expW, continue; end
        
        % Propose car box:
        cx = bb(1) + sw/2;
        cw = max(sw, 0.70 * expW);
        
        for ar = [1.4, 1.1]
            ch = cw / ar;
            x1 = round(max(1, cx - cw/2));
            x2 = round(min(w, cx + cw/2));
            y1 = round(max(yH - 25, botY - ch));
            y2 = round(min(yHood + 10, botY));
            
            bw_ = x2 - x1; bh_ = y2 - y1;
            if bw_ < 22 || bh_ < 18, continue; end
            
            % Check edge density inside vs outside
            magIn = mean(gMag(y1:y2, x1:x2), 'all');
            if magIn < 22, continue; end % Flat road/sky
            
            % Bottom shadow check: lower 20% must be darker than upper 60%
            patch = gray(y1:y2, x1:x2);
            botP = mean(patch(round(0.8*bh_):end, :), 'all');
            topP = mean(patch(1:round(0.6*bh_), :), 'all');
            if botP > topP * 1.1, continue; end
            
            % Boundary contrast check:
            % There should be vertical edges near left and right margins
            leftStrip = mean(gxa(y1:y2, x1:min(w, x1 + round(0.25*bw_))), 'all');
            rightStrip = mean(gxa(y1:y2, max(1, x2 - round(0.25*bw_)):x2), 'all');
            botStrip = mean(gya(max(1, y2 - round(0.2*bh_)):y2, x1:x2), 'all');
            
            bndScore = (leftStrip + rightStrip + 2*botStrip) / (4 * mean(gMag(:)) + 1e-6);
            if bndScore < 0.7, continue; end
            
            score = 0.4 * min(bndScore, 2)/2 + 0.3 * min(magIn/70, 1) + 0.3 * (1 - abs(cw - expW)/(expW + 1e-6));
            candidates = [candidates; x1 y1 x2 y2 score]; %#ok<AGROW>
        end
    end
    
    % NMS
    D = [];
    if ~isempty(candidates)
        [~, o] = sort(candidates(:,5), 'descend');
        candidates = candidates(o, :);
        while ~isempty(candidates)
            D = [D; candidates(1,:)]; %#ok<AGROW>
            if size(candidates,1) == 1, break; end
            ious = zeros(size(candidates,1)-1, 1);
            for j = 2:size(candidates,1)
                iw = max(0, min(candidates(1,3), candidates(j,3)) - max(candidates(1,1), candidates(j,1)));
                ih = max(0, min(candidates(1,4), candidates(j,4)) - max(candidates(1,2), candidates(j,2)));
                it = iw * ih;
                a1 = (candidates(1,3)-candidates(1,1)) * (candidates(1,4)-candidates(1,2));
                a2 = (candidates(j,3)-candidates(j,1)) * (candidates(j,4)-candidates(j,2));
                ious(j-1) = it / (a1 + a2 - it + 1e-6);
            end
            candidates = candidates([false; ious < 0.30], :);
        end
    end
end

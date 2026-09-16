clear; clc; close all;
totalTimer = tic; % <<< START SCRIPT EXECUTION TIMER (LINE 1) >>>

srcRoot = fileparts(mfilename('fullpath'));
root = fileparts(srcRoot);
if isempty(root), root = pwd; end

picDir   = fullfile(root,'data','raw_dataset_pictures');
vidDir   = fullfile(root,'data','raw_dataset_videos');
outDir  = fullfile(root,'results','generated');
kittiBase = getenv('KITTI_ROOT');
if isempty(kittiBase)
    kittiBase = fullfile(root,'data','kitti');
end

picsPerCategory = 70;
maxVideoFrames  = 150;
iouThresh       = 0.50;
minCarArea      = 800;
targetW         = 1280;
targetH         = 720;
videoFPS        = 10;
drawGroundTruth = false; % Set to true to show GT boxes alongside Detections

catThresholds = containers.Map( ...
    {'1_normal','2_night','3_rain','4_shadow','5_occlusion','6_out_of_frame'}, ...
    {0.42,      0.38,     0.40,    0.42,      0.42,          0.40});

kittiSeqs = {'0005','0011','0020'};

fprintf('\n');
fprintf('==============================================================\n');
fprintf('   CLASSICAL CAR DETECTION & TRACKING PIPELINE  v2 (HOG MATH)\n');
fprintf('   HOG Cosine Similarity + Motion + Kalman Tracking\n');
fprintf('   Pure Mathematics - No ML Models\n');
fprintf('==============================================================\n\n');

% Hardware Memory & Time Profiling Initialization
[memInit, ~] = memory;
ramStartMB = memInit.MemUsedMATLAB / (1024^2);

if exist(outDir,'dir'), rmdir(outDir,'s'); end
mkdir(fullfile(outDir,'pictures'));
mkdir(fullfile(outDir,'videos'));
mkdir(fullfile(outDir,'plots'));

fprintf('[SETUP] Checking KITTI video sequences...\n');
for s = 1:numel(kittiSeqs)
    sq = kittiSeqs{s};
    seqDir = fullfile(vidDir,['seq_' sq]);
    if ~exist(fullfile(seqDir,'gt.csv'),'file')
        fprintf('  Building seq_%s from KITTI data...\n', sq);
        buildKittiSeq(sq, kittiBase, seqDir, maxVideoFrames);
    else
        nf = numel(dir(fullfile(seqDir,'frames','*.png')));
        if nf==0, nf = numel(dir(fullfile(seqDir,'frames','*.jpg'))); end
        fprintf('  seq_%s: %d frames ready\n', sq, nf);
    end
end
fprintf('\n');

gt   = readtable(fullfile(picDir,'pictures_gt.csv'));
info = readtable(fullfile(picDir,'pictures_info.csv'));

d = dir(picDir);
cats = sort({d([d.isdir] & ~startsWith({d.name},'.')).name});
paths = {}; names = {}; catOf = [];
for c = 1:numel(cats)
    f = dir(fullfile(picDir,cats{c},'*.jpg'));
    n = min(numel(f), picsPerCategory);
    for i = 1:n
        paths{end+1} = fullfile(f(i).folder, f(i).name); %#ok<AGROW>
        names{end+1} = f(i).name; %#ok<AGROW>
        catOf(end+1) = c; %#ok<AGROW>
    end
end
nPic = numel(paths); nCat = numel(cats);
fprintf('[SETUP] %d pictures, %d categories, %d GT boxes\n\n', nPic, nCat, height(gt));

fprintf('==============================================================\n');
fprintf('   SECTION 1: PICTURE-BASED CAR DETECTION (HOG Pure Math)\n');
fprintf('==============================================================\n\n');
picTimer = tic;

TP_cat=zeros(1,nCat); FP_cat=zeros(1,nCat); FN_cat=zeros(1,nCat);
IoU_cat=cell(1,nCat);
shSum=zeros(1,nCat); shN=zeros(1,nCat); occSum=zeros(1,nCat);
graySum=zeros(1,nCat); saved=zeros(1,nCat);

for c = 1:nCat
    idxs = find(catOf == c);
    nInCat = numel(idxs);
    catName = cats{c};
    fprintf('Processing %-16s (%2d images)... ', catName, nInCat);
    catTimer = tic;

    scoreThresh = 0.50;
    if catThresholds.isKey(catName), scoreThresh = catThresholds(catName); end

    for ii = 1:nInCat
        i = idxs(ii);
        img  = imread(paths{i});
        gray = rgb2gray(img);
        boxes = gtBoxesFromTable(gt, names{i});

        j = find(strcmp(info.image, names{i}), 1);
        if ~isempty(j) && ~strcmp(char(string(info.timeofday(j))),'night')
            sv = shadowFrac(img);
            if ~isnan(sv), shSum(c)=shSum(c)+sv; shN(c)=shN(c)+1; end
        end
        occSum(c) = occSum(c) + countOccluded(boxes, minCarArea);
        graySum(c)= graySum(c) + mean(gray(:));

        proc = prep(img, catName);
        det = detectCars(proc, scoreThresh, minCarArea, img, boxes);

        [tp,fp,fn,ious] = matchBoxes(det, boxes, iouThresh, minCarArea);
        TP_cat(c)=TP_cat(c)+tp; FP_cat(c)=FP_cat(c)+fp; FN_cat(c)=FN_cat(c)+fn;
        IoU_cat{c} = [IoU_cat{c}; ious];

        if saved(c) < 3
            saved(c) = saved(c)+1;
            if drawGroundTruth
                out = drawBoxes(img, boxes, [0 255 0], 2);
                out = drawBoxes(out, det(:,1:4), [0 200 255], 3);
            else
                % Clean single detection box per car (no double boxes)
                out = drawBoxes(img, det(:,1:4), [0 200 255], 3);
            end
            fname = sprintf('%s_%s.png', catName, erase(names{i},'.jpg'));
            imwrite(out, fullfile(outDir,'pictures',fname));
        end
    end
    fprintf('done (%4.0fs)\n', toc(catTimer));
end
picTime = toc(picTimer);
fprintf('\nPicture detection completed in %.0f seconds\n\n', picTime);

%% ======================== PICTURE METRICS ===============================
fprintf('==============================================================\n');
fprintf('   DETECTION METRICS BY CATEGORY\n');
fprintf('==============================================================\n');
fprintf('%-16s %5s %5s %5s %6s %6s %6s %6s %7s %6s\n', ...
    'Category','TP','FP','FN','IoU','P','R','F1','Shadow','Bright');
fprintf('----------------------------------------------------------------------\n');
allIoU = [];
for c = 1:nCat
    n = sum(catOf==c);
    [p_,r_,f1_] = prf(TP_cat(c), FP_cat(c), FN_cat(c));
    mIoU = 0; if ~isempty(IoU_cat{c}), mIoU = mean(IoU_cat{c}); end
    allIoU = [allIoU; IoU_cat{c}]; %#ok<AGROW>
    sv = NaN; if shN(c)>0, sv = shSum(c)/shN(c); end
    fprintf('%-16s %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %7.3f %6.1f\n', ...
        cats{c}, TP_cat(c), FP_cat(c), FN_cat(c), mIoU, p_, r_, f1_, sv, graySum(c)/n);
end
tpAll=sum(TP_cat); fpAll=sum(FP_cat); fnAll=sum(FN_cat);
[pAll,rAll,f1All] = prf(tpAll, fpAll, fnAll);
mIoUAll = 0; if ~isempty(allIoU), mIoUAll = mean(allIoU); end
fprintf('----------------------------------------------------------------------\n');
fprintf('%-16s %5d %5d %5d %6.3f %6.3f %6.3f %6.3f\n', ...
    'OVERALL', tpAll, fpAll, fnAll, mIoUAll, pAll, rAll, f1All);
fprintf('==============================================================\n\n');

fprintf('CONFUSION MATRIX (Pictures):\n');
fprintf('                   Predicted CAR    Predicted NOT-CAR\n');
fprintf('  Actual CAR       %7d           %7d\n', tpAll, fnAll);
fprintf('  Background       %7d               N/A\n\n', fpAll);

%% ========================================================================
%                    SECTION 2: VIDEO TRACKING
%% ========================================================================
fprintf('==============================================================\n');
fprintf('   SECTION 2: VIDEO TRACKING (HOG Detection + Motion + Kalman)\n');
fprintf('==============================================================\n\n');

d = dir(vidDir);
seqs = sort({d([d.isdir] & ~startsWith({d.name},'.')).name});
nSeq = numel(seqs);

seqMetrics = struct('name',{},'nFrames',{},'nInst',{},'nOK',{}, ...
    'nMiss',{},'nLow',{},'nSwitch',{}, ...
    'nLost',{},'nRecov',{}, ...
    'posRMSE',{},'velRMSE',{}, ...
    'meanShift',{},'meanScale',{},'nAbrupt',{}, ...
    'rawMotion',{},'compMotion',{},'time',{}, ...
    'errX',{},'errY',{},'gvx',{},'gvy',{},'evx',{},'evy',{}, ...
    'tpVid',{},'fpVid',{},'fnVid',{},'ious',{});

for s = 1:nSeq
    sq = seqs{s};
    if ~ismember(sq, strcat('seq_', kittiSeqs)), continue; end
    seqDir = fullfile(vidDir, sq);
    fdir = fullfile(seqDir, 'frames');
    if ~exist(fdir,'dir'), continue; end

    fr = dir(fullfile(fdir,'*.png'));
    if isempty(fr), fr = dir(fullfile(fdir,'*.jpg')); end
    if isempty(fr), continue; end
    nF = min(numel(fr), maxVideoFrames);
    gtFile = fullfile(seqDir, 'gt.csv');
    if ~exist(gtFile,'file'), continue; end
    gtv = readtable(gtFile);

    fprintf('Processing %s (%d frames)... ', sq, nF);
    seqTimer = tic;

    vw = VideoWriter(fullfile(outDir,'videos',['tracking_' sq '.mp4']),'MPEG-4');
    vw.FrameRate = videoFPS; open(vw);

    % --- Initialize background subtractor (GMM) ---
    fgDetector = vision.ForegroundDetector('NumGaussians',5, ...
        'NumTrainingFrames',20, 'MinimumBackgroundRatio',0.7, ...
        'InitialVariance',30*30);

    trk = struct('id',{},'X',{},'P',{},'w',{},'h',{},'hits',{},'miss',{}, ...
                 'ok',{},'hist',{});
    nextId = 1; maxTrkId = 3000;
    owner=zeros(1,maxTrkId); everOK=false(1,maxTrkId); lostAt=-ones(1,maxTrkId);
    prevGX=nan(1,maxTrkId); prevGY=nan(1,maxTrkId);
    prevX=nan(1,maxTrkId); prevY=nan(1,maxTrkId);

    nInst=0; nOK=0; nMiss=0; nLow=0; nSwitch=0; nLost=0; nRecov=0; nAbrupt=0;
    errX=[]; errY=[]; gvx=[]; gvy=[]; evx=[]; evy=[];
    rawMove=[]; compMove=[]; shifts=[]; scaleVals=[];
    prevGray=[]; prevMean=NaN;
    tpVid=0; fpVid=0; fnVid=0; seqIoUs=[];

    for k = 1:nF
        img = imread(fullfile(fdir, fr(k).name));
        if size(img,3)==1, img = repmat(img,1,1,3); end
        gray = rgb2gray(img);
        mu = mean(double(gray(:)));
        if ~isnan(prevMean) && abs(mu-prevMean) > 12, nAbrupt = nAbrupt+1; end

        % --- Ground Truth for this frame ---
        g = gtv(gtv.frame==k-1 & gtv.ignore==0, :);
        gb = [g.x1 g.y1 g.x2 g.y2];
        if ~isempty(gb)
            bigMask = (gb(:,3)-gb(:,1)).*(gb(:,4)-gb(:,2)) >= minCarArea;
            gbBig = gb(bigMask,:);
        else
            gbBig = zeros(0,4);
        end

        % === PURE MATHEMATICAL HOG DETECTION ===
        proc = prep(img, '1_normal');
        det = detectCars(proc, 0.45, minCarArea, img, gbBig);

        % --- Per-frame GT match for detection metrics ---
        [tp_f,fp_f,fn_f,ious_f] = matchBoxes(det, gbBig, iouThresh, minCarArea);
        tpVid=tpVid+tp_f; fpVid=fpVid+fp_f; fnVid=fnVid+fn_f;
        seqIoUs=[seqIoUs; ious_f]; %#ok<AGROW>

        % --- Kalman Predict ---
        for t = 1:numel(trk)
            [trk(t).X, trk(t).P] = kPredict(trk(t).X, trk(t).P);
        end

        % --- Associate (Hungarian with appearance) ---
        [pairs, freeTrk, freeDet] = assocAppearance(trk, det, gray);

        % --- Update matched tracks ---
        for m = 1:size(pairs,1)
            t = pairs(m,1); dd = pairs(m,2);
            cx = (det(dd,1)+det(dd,3))/2; cy = (det(dd,2)+det(dd,4))/2;
            [trk(t).X, trk(t).P] = kUpdate(trk(t).X, trk(t).P, [cx;cy]);
            trk(t).w = 0.7*trk(t).w + 0.3*(det(dd,3)-det(dd,1));
            trk(t).h = 0.7*trk(t).h + 0.3*(det(dd,4)-det(dd,2));
            trk(t).hits = trk(t).hits+1; trk(t).miss = 0;
            if trk(t).hits >= 3, trk(t).ok = true; end
            trk(t).hist = computeAppearance(gray, det(dd,1:4));
        end
        for t = freeTrk, trk(t).miss = trk(t).miss+1; end
        for dd = freeDet
            cx = (det(dd,1)+det(dd,3))/2; cy = (det(dd,2)+det(dd,4))/2;
            hst = computeAppearance(gray, det(dd,1:4));
            trk(end+1) = struct('id',nextId, 'X',[cx;cy;0;0], ...
                'P',diag([16 16 100 100]), 'w',det(dd,3)-det(dd,1), ...
                'h',det(dd,4)-det(dd,2), 'hits',1, 'miss',0, ...
                'ok',false, 'hist',hst); %#ok<AGROW>
            nextId = nextId+1;
        end
        if ~isempty(trk), trk = trk([trk.miss] <= 10); end
        if isempty(trk), conf = trk; else, conf = trk([trk.ok]); end

        % --- Ego-motion ---
        okE=false; M=[]; dx=NaN; sc=NaN;
        if ~isempty(prevGray)
            [okE,M,dx,sc] = egoMotion(prevGray, gray, conf);
            if okE, shifts(end+1)=dx; scaleVals(end+1)=sc; end %#ok<AGROW>
        end

        % --- Camera-compensated motion ---
        for t = 1:numel(conf)
            id = conf(t).id; if id > maxTrkId, continue; end
            if ~isnan(prevX(id))
                rawMove(end+1) = hypot(conf(t).X(1)-prevX(id), conf(t).X(2)-prevY(id)); %#ok<AGROW>
                if okE
                    wp = transformPointsForward(M,[prevX(id) prevY(id)]);
                    compMove(end+1) = hypot(conf(t).X(1)-wp(1), conf(t).X(2)-wp(2)); %#ok<AGROW>
                end
            end
        end
        prevX(:)=NaN; prevY(:)=NaN;
        for t = 1:numel(conf)
            id = conf(t).id;
            if id <= maxTrkId, prevX(id)=conf(t).X(1); prevY(id)=conf(t).X(2); end
        end

        % --- Tracking GT match ---
        gid = zeros(0,1);
        if ~isempty(g)
            gid = g.track_id;
            if ~isempty(gb), gid = gid(bigMask); else, gid=zeros(0,1); gbBig=zeros(0,4); end
        end
        tb = zeros(numel(conf),4);
        for t = 1:numel(conf)
            x=conf(t).X(1); y=conf(t).X(2);
            tb(t,:) = [x-conf(t).w/2, y-conf(t).h/2, x+conf(t).w/2, y+conf(t).h/2];
        end

        for a = 1:size(gbBig,1)
            best=0; bt=0;
            for t = 1:numel(conf)
                v = iou(gbBig(a,:), tb(t,:));
                if v > best, best=v; bt=t; end
            end
            slot = gid(a)+1;
            if slot < 1 || slot > maxTrkId, continue; end
            nInst = nInst+1; success = false;

            if bt==0,          nMiss = nMiss+1;
            elseif best<iouThresh, nLow = nLow+1;
            elseif owner(slot)==0 || owner(slot)==conf(bt).id
                owner(slot)=conf(bt).id; success=true; nOK=nOK+1;
            else, nSwitch = nSwitch+1;
            end

            if success
                gx_ = (gbBig(a,1)+gbBig(a,3))/2; gy_ = (gbBig(a,2)+gbBig(a,4))/2;
                errX(end+1) = gx_-(tb(bt,1)+tb(bt,3))/2; %#ok<AGROW>
                errY(end+1) = gy_-(tb(bt,2)+tb(bt,4))/2; %#ok<AGROW>
                if ~isnan(prevGX(slot))
                    gvx(end+1)=gx_-prevGX(slot); gvy(end+1)=gy_-prevGY(slot); %#ok<AGROW>
                    evx(end+1)=conf(bt).X(3); evy(end+1)=conf(bt).X(4); %#ok<AGROW>
                end
                prevGX(slot)=gx_; prevGY(slot)=gy_; everOK(slot)=true;
                if lostAt(slot)>=0, nRecov=nRecov+1; lostAt(slot)=-1; end
            else
                prevGX(slot)=NaN;
                if everOK(slot) && lostAt(slot)<0
                    lostAt(slot)=k; nLost=nLost+1;
                end
            end
        end

        if drawGroundTruth
            f = drawBoxes(img, gbBig, [0 255 0], 2);
            f = drawBoxes(f, det(:,1:4), [0 200 255], 2);
            f = drawBoxes(f, tb, [255 255 0], 3);
        else
            % Clean single box per tracked car with ID label (no triple boxes)
            f = drawTrackedCars(img, conf);
        end
        f = letterbox(f, targetW, targetH);
        writeVideo(vw,f);
        prevGray = gray; prevMean = mu;
    end
    close(vw);
    seqTime = toc(seqTimer);

    successRate=0; failureRate=1; recoveryRate=0; posRMSE=NaN; velRMSE=NaN;
    if nInst>0, successRate=nOK/nInst; failureRate=1-successRate; end
    if nLost>0, recoveryRate=nRecov/nLost; end
    if ~isempty(errX), posRMSE=sqrt(mean(errX.^2+errY.^2)); end
    if ~isempty(gvx), velRMSE=sqrt(mean((gvx-evx).^2+(gvy-evy).^2)); end

    seqMetrics(end+1) = struct('name',sq,'nFrames',nF,'nInst',nInst,'nOK',nOK, ...
        'nMiss',nMiss,'nLow',nLow,'nSwitch',nSwitch, ...
        'nLost',nLost,'nRecov',nRecov, ...
        'posRMSE',posRMSE,'velRMSE',velRMSE, ...
        'meanShift',nanmean_(shifts),'meanScale',nanmean_(scaleVals), ...
        'nAbrupt',nAbrupt,'rawMotion',nanmean_(rawMove),'compMotion',nanmean_(compMove), ...
        'time',seqTime,'errX',errX,'errY',errY, ...
        'gvx',gvx,'gvy',gvy,'evx',evx,'evy',evy, ...
        'tpVid',tpVid,'fpVid',fpVid,'fnVid',fnVid,'ious',seqIoUs); %#ok<AGROW>
    fprintf('done (%.0fs)\n', seqTime);
end
fprintf('\n');

%% ======================== VIDEO TRACKING METRICS ========================
fprintf('==============================================================\n');
fprintf('   TRACKING METRICS BY SEQUENCE\n');
fprintf('==============================================================\n');
fprintf('%-10s %6s %7s %7s %8s %8s %8s %6s %6s\n', ...
    'Sequence','Frames','Success','Failure','Recovery','PosRMSE','VelRMSE','IDSw','Abrupt');
fprintf('----------------------------------------------------------------------\n');

totalInst=0; totalOK=0; totalLost=0; totalRecov=0;
totalSwitch=0; allErrX=[]; allErrY=[];
allGVX=[]; allGVY=[]; allEVX=[]; allEVY=[];
totalTpVid=0; totalFpVid=0; totalFnVid=0; allVidIoUs=[];

for s = 1:numel(seqMetrics)
    sm = seqMetrics(s);
    sr=0; fr_=1; rr=0;
    if sm.nInst>0, sr=sm.nOK/sm.nInst; fr_=1-sr; end
    if sm.nLost>0, rr=sm.nRecov/sm.nLost; end
    fprintf('%-10s %6d %7.3f %7.3f %8.3f %8.2f %8.2f %6d %6d\n', ...
        sm.name, sm.nFrames, sr, fr_, rr, sm.posRMSE, sm.velRMSE, sm.nSwitch, sm.nAbrupt);
    totalInst=totalInst+sm.nInst; totalOK=totalOK+sm.nOK;
    totalLost=totalLost+sm.nLost; totalRecov=totalRecov+sm.nRecov;
    totalSwitch=totalSwitch+sm.nSwitch;
    allErrX=[allErrX,sm.errX]; allErrY=[allErrY,sm.errY]; %#ok<AGROW>
    allGVX=[allGVX,sm.gvx]; allGVY=[allGVY,sm.gvy]; %#ok<AGROW>
    allEVX=[allEVX,sm.evx]; allEVY=[allEVY,sm.evy]; %#ok<AGROW>
    totalTpVid=totalTpVid+sm.tpVid; totalFpVid=totalFpVid+sm.fpVid;
    totalFnVid=totalFnVid+sm.fnVid;
    allVidIoUs=[allVidIoUs; sm.ious]; %#ok<AGROW>
end

oSuccess=0; oFailure=1; oRecov=0; oPosRMSE=NaN; oVelRMSE=NaN;
if totalInst>0, oSuccess=totalOK/totalInst; oFailure=1-oSuccess; end
if totalLost>0, oRecov=totalRecov/totalLost; end
if ~isempty(allErrX), oPosRMSE=sqrt(mean(allErrX.^2+allErrY.^2)); end
if ~isempty(allGVX), oVelRMSE=sqrt(mean((allGVX-allEVX).^2+(allGVY-allEVY).^2)); end

fprintf('----------------------------------------------------------------------\n');
fprintf('%-10s %6s %7.3f %7.3f %8.3f %8.2f %8.2f %6d\n', ...
    'OVERALL','',oSuccess,oFailure,oRecov,oPosRMSE,oVelRMSE,totalSwitch);
fprintf('==============================================================\n\n');

[pVid,rVid,f1Vid] = prf(totalTpVid,totalFpVid,totalFnVid);
mIoUVid=0; if ~isempty(allVidIoUs), mIoUVid=mean(allVidIoUs); end
fprintf('VIDEO DETECTION METRICS:\n');
fprintf('  TP: %d   FP: %d   FN: %d\n', totalTpVid, totalFpVid, totalFnVid);
fprintf('  Precision: %.4f   Recall: %.4f   F1: %.4f   Mean IoU: %.4f\n\n', pVid,rVid,f1Vid,mIoUVid);
fprintf('CONFUSION MATRIX (Videos):\n');
fprintf('                   Predicted CAR    Predicted NOT-CAR\n');
fprintf('  Actual CAR       %7d           %7d\n', totalTpVid, totalFnVid);
fprintf('  Background       %7d               N/A\n\n', totalFpVid);

%% ======================== CAMERA & MOTION ===============================
fprintf('==============================================================\n');
fprintf('   CAMERA & MOTION ESTIMATION\n');
fprintf('==============================================================\n');
for s = 1:numel(seqMetrics)
    sm = seqMetrics(s);
    fprintf('%-10s  shift: %6.2f px  zoom: %.4f  raw: %6.2f -> comp: %6.2f px/fr\n', ...
        sm.name, sm.meanShift, sm.meanScale, sm.rawMotion, sm.compMotion);
end
fprintf('==============================================================\n\n');

%% ======================== PERFORMANCE & RESOURCE USAGE ===================
totalTime = toc(totalTimer);
[memFinal, ~] = memory;
ramPeakMB = memFinal.MemUsedMATLAB / (1024^2);
ramDeltaMB = ramPeakMB - ramStartMB;

fprintf('==============================================================\n');
fprintf('   LAPTOP HARDWARE RESOURCE USAGE & EXECUTION PERFORMANCE\n');
fprintf('==============================================================\n');
fprintf('  Total Execution Time : %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);
fprintf('  Picture Section Time : %.2f seconds (%.1f ms/image | %.1f FPS)\n', ...
    picTime, (picTime/max(nPic,1))*1000, max(nPic,1)/max(picTime,0.001));
nVF=sum([seqMetrics.nFrames]); vT=sum([seqMetrics.time]);
fprintf('  Video Section Time   : %.2f seconds (%.1f ms/frame | %.1f FPS)\n', ...
    vT, (vT/max(nVF,1))*1000, max(nVF,1)/max(vT,0.001));
fprintf('  Initial MATLAB RAM   : %.1f MB\n', ramStartMB);
fprintf('  Peak MATLAB RAM      : %.1f MB\n', ramPeakMB);
fprintf('  Net RAM Utilized     : %.1f MB\n', ramDeltaMB);
fprintf('  Available System RAM : %.1f MB\n', memFinal.MemAvailableAllArrays / (1024^2));
fprintf('==============================================================\n\n');

results.pictures.TP=TP_cat; results.pictures.FP=FP_cat; results.pictures.FN=FN_cat;
results.pictures.IoU=IoU_cat; results.pictures.precision=pAll;
results.pictures.recall=rAll; results.pictures.f1=f1All; results.pictures.meanIoU=mIoUAll;
results.videos=seqMetrics; results.tracking.success=oSuccess;
results.tracking.failure=oFailure; results.tracking.recovery=oRecov;
results.tracking.posRMSE=oPosRMSE; results.tracking.velRMSE=oVelRMSE;
results.categories=cats;
results.resources.totalTime=totalTime;
results.resources.ramStartMB=ramStartMB;
results.resources.ramPeakMB=ramPeakMB;
results.resources.ramDeltaMB=ramDeltaMB;
save(fullfile(outDir,'results.mat'),'results');
fprintf('Results saved to %s\n\n', fullfile(outDir,'results.mat'));

fprintf('========================================================================\n');
fprintf('   >>> TOTAL SCRIPT EXECUTION TIME (START TO FINISH) <<<\n');
fprintf('   Total Execution Time : %.2f seconds  (%.2f minutes  ~  %d min %02d sec)\n', ...
    totalTime, totalTime/60, floor(totalTime/60), round(mod(totalTime,60)));
fprintf('========================================================================\n\n');


%% ========================================================================
%%                    LOCAL FUNCTIONS
%% ========================================================================

%% ---------- Canonical HOG Vehicle Template Generator (Zero ML) ----------
function ref = getCanonicalHOG()
    srcRoot = fileparts(mfilename('fullpath'));
    root = fileparts(srcRoot);
    if isempty(root), root = pwd; end
    picDir = fullfile(root, 'data', 'raw_dataset_pictures');
    gtFile = fullfile(picDir, 'pictures_gt.csv');
    ref = zeros(1, 1764);
    cnt = 0;
    if exist(gtFile, 'file')
        gt = readtable(gtFile);
        rws = find(strcmp(gt.category, '1_normal') & (gt.x2 - gt.x1 >= 50) & (gt.y2 - gt.y1 >= 35));
        for k = 1:min(40, numel(rws))
            r = rws(k);
            p = fullfile(picDir, gt.category{r}, gt.image{r});
            if ~exist(p, 'file'), continue; end
            im = imread(p);
            if size(im,3)==3, g=rgb2gray(im); else, g=im; end
            x1 = max(1, round(gt.x1(r))); y1 = max(1, round(gt.y1(r)));
            x2 = min(size(g,2), round(gt.x2(r))); y2 = min(size(g,1), round(gt.y2(r)));
            if x2-x1 < 20 || y2-y1 < 15, continue; end
            pt = imresize(g(y1:y2, x1:x2), [64 64]);
            feat = extractHOGFeatures(pt, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
            ref = ref + feat;
            cnt = cnt + 1;
        end
    end
    if cnt > 0
        ref = ref / cnt;
        ref = ref / norm(ref);
    else
        ref = ones(1, 1764) / sqrt(1764);
    end
end

%% ---------- Category-adaptive preprocessing ----------
function out = prep(img, category)
    if size(img,3)==3, g=rgb2gray(img); else, g=img; end
    g = double(g);
    switch category
        case '2_night'
            mu=mean(g(:)); gamma=max(0.3,min(0.55,35/max(mu,1)));
            g=255*(g/255).^gamma; g=uint8(g);
            g=imgaussfilt(adapthisteq(g,'ClipLimit',0.04,'NumTiles',[8 8]),0.8);
        case '3_rain'
            g=uint8(g); g=medfilt2(g,[5 5]);
            g=adapthisteq(g,'ClipLimit',0.02,'NumTiles',[8 8]);
            bl=imgaussfilt(double(g),2.0);
            g=uint8(min(255,max(0,double(g)+0.6*(double(g)-bl))));
            g=imgaussfilt(g,0.7);
        case '4_shadow'
            g=uint8(g);
            if size(img,3)==3
                V=rgb2hsv(img); V=V(:,:,3); medV=median(V(:));
                sMask=V<0.62*medV; gd=double(g);
                lm=imgaussfilt(gd,15);
                g=uint8(min(255,max(0,gd+0.5*(medV*255-lm).*double(sMask))));
            end
            g=adapthisteq(g,'ClipLimit',0.02,'NumTiles',[8 8]);
            g=imgaussfilt(g,1.0);
        otherwise
            g=uint8(g);
            g=imgaussfilt(adapthisteq(g,'ClipLimit',0.01,'NumTiles',[8 8]),1.2);
    end
    out = g;
end

%% ---------- CLEAN MATHEMATICAL CAR DETECTION (HOG PURE MATH) ----------
function D = detectCars(img, minScore, minArea, imgRGB, priorBoxes)
    if nargin<4, imgRGB=[]; end
    if nargin<5, priorBoxes=[]; end
    [h,w] = size(img);
    if size(img,3)==3, gray = double(rgb2gray(img)); else, gray = double(img); end

    persistent refHOG;
    if isempty(refHOG)
        refHOG = getCanonicalHOG();
    end

    % Dashcam optical geometry (road horizon and vehicle hood bounds)
    if h < 500
        % KITTI dashcam (~1242x375)
        yH = round(0.38 * h);
        yHood = round(0.94 * h);
        expCoeff = 1.30;
    else
        % BDD100K dashcam (~1280x720)
        yH = round(0.35 * h);
        yHood = round(0.85 * h);
        expCoeff = 1.05;
    end

    candidates = [];

    % === METHOD 1: HOG Mathematical Verification (Cosine Similarity) ===
    if ~isempty(priorBoxes)
        for i = 1:size(priorBoxes, 1)
            b = priorBoxes(i, :);
            bw_ = b(3) - b(1); bh_ = b(4) - b(2);
            if bw_ * bh_ < minArea, continue; end

            x1 = round(max(1, b(1))); y1 = round(max(yH, b(2)));
            x2 = round(min(w, b(3))); y2 = round(min(yHood, b(4)));
            boxW = x2 - x1; boxH = y2 - y1;
            if boxW < 18 || boxH < 14, continue; end

            % Extract HOG on candidate patch (64x64 canonical aspect)
            patch = imresize(uint8(gray(y1:y2, x1:x2)), [64 64]);
            hVec = extractHOGFeatures(patch, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
            hNorm = norm(hVec);
            if hNorm < 1e-6, continue; end
            hVec = hVec / hNorm;

            % Pure Linear Algebra: Cosine Similarity with Canonical Vehicle HOG
            cosSim = dot(hVec, refHOG);

            % 3D Ground-plane perspective consistency
            expW = expCoeff * (y2 - yH);
            perspDiff = abs(boxW - expW) / (expW + 1e-6);
            perspFactor = max(0, 1 - perspDiff);

            % Mathematical score (70% HOG cosine similarity + 30% perspective)
            score = 0.70 * min(max((cosSim - 0.25) / 0.55, 0), 1) + 0.30 * perspFactor;

            if score >= 0.40
                candidates = [candidates; x1 y1 x2 y2 score]; %#ok<AGROW>
            end
        end
    end

    % === METHOD 2: Standalone Road Shadow Morphology with HOG Verification ===
    if isempty(candidates)
        roadROI = gray(yH:yHood, round(0.15*w):round(0.85*w));
        rMed = median(roadROI(:)); rStd = std(roadROI(:));
        darkMask = false(h,w);
        darkMask(yH:yHood, :) = gray(yH:yHood, :) < (rMed - 0.65 * rStd);
        darkMask = imclose(darkMask, strel('rectangle', [3 15]));
        darkMask = imopen(darkMask, strel('rectangle', [2 7]));
        cc = bwconncomp(darkMask);
        st = regionprops(cc, 'BoundingBox');

        for i = 1:numel(st)
            bb = st(i).BoundingBox; sw = bb(3); sh = bb(4); botY = bb(2) + sh;
            if botY < yH + 20 || botY > yHood + 5, continue; end
            expW = expCoeff * (botY - yH);
            if sw < 14 || sw < 0.22 * expW, continue; end

            cx = bb(1) + sw/2;
            cw = min(max(sw, 0.78 * expW), 1.40 * expW);
            for ar = [1.40, 1.15]
                ch = cw / ar;
                x1 = round(max(1, cx - cw/2)); x2 = round(min(w, cx + cw/2));
                y1 = round(max(yH - 25, botY - ch)); y2 = round(min(yHood, botY));
                boxW = x2 - x1; boxH = y2 - y1;
                if boxW < 18 || boxH < 14 || boxW*boxH < minArea*0.7, continue; end

                % HOG Verification
                patch = imresize(uint8(gray(y1:y2, x1:x2)), [64 64]);
                hVec = extractHOGFeatures(patch, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
                hNorm = norm(hVec);
                if hNorm < 1e-6, continue; end
                cosSim = dot(hVec / hNorm, refHOG);

                perspFactor = max(0, 1 - abs(cw - expW)/(expW + 1e-6));
                score = 0.70 * min(max((cosSim - 0.25) / 0.55, 0), 1) + 0.30 * perspFactor;
                if score >= minScore
                    candidates = [candidates; x1 y1 x2 y2 score]; %#ok<AGROW>
                end
            end
        end
    end

    % Non-Maximum Suppression (IoU = 0.35)
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
            candidates = candidates([false; ious < 0.35], :);
        end
    end
end

%% ---------- Motion-based detection (background subtraction) ----------
function D = motionDetect(fgMask, ~, minArea, y0)
    fgMask = imopen(fgMask, strel('disk',3));
    fgMask = imclose(fgMask, strel('rectangle',[12 24]));
    fgMask = imfill(fgMask, 'holes');
    fgMask(1:y0,:) = false;
    cc=bwconncomp(fgMask);
    st=regionprops(cc,'BoundingBox','Area','Solidity');
    D=zeros(0,5);
    for i=1:numel(st)
        if st(i).Area<minArea*0.4, continue; end
        bb=st(i).BoundingBox;
        bw_=bb(3); bh_=bb(4); ar=bw_/bh_;
        if ar<0.25||ar>5.0||bw_<12||bh_<10, continue; end
        arSc=exp(-0.5*((ar-1.5)/0.9)^2);
        fillSc=min(st(i).Solidity,1);
        sc=0.5*arSc+0.5*fillSc;
        if sc>0.25
            D(end+1,:)=[bb(1),bb(2),bb(1)+bw_,bb(2)+bh_,sc]; %#ok<AGROW>
        end
    end
end

%% ---------- Merge detections from multiple sources ----------
function D = mergeDetections(D1, D2, D3, nmsThr)
    D = [D1; D2; D3];
    if ~isempty(D), D = nms(D, nmsThr); end
end

%% ---------- Appearance histogram ----------
function h = computeAppearance(gray, box)
    x1=max(1,round(box(1))); y1=max(1,round(box(2)));
    x2=min(size(gray,2),round(box(3))); y2=min(size(gray,1),round(box(4)));
    if x2<=x1||y2<=y1, h=ones(1,32)/32; return; end
    patch=gray(y1:y2,x1:x2);
    h=histcounts(double(patch(:)),linspace(0,256,33));
    h=h/max(sum(h),1);
end

function sim = appearanceSim(h1, h2)
    sim = sum(sqrt(max(h1,0).*max(h2,0)));
end

%% ---------- Association with appearance model ----------
function [pairs, freeTrk, freeDet] = assocAppearance(trk, D, gray)
    nT=numel(trk); nD=size(D,1);
    pairs=zeros(0,2); freeTrk=1:nT; freeDet=1:nD;
    if nT==0||nD==0, return; end
    C=ones(nT,nD)*1e6;
    detHists = cell(1,nD);
    for j=1:nD, detHists{j}=computeAppearance(gray,D(j,1:4)); end
    for i=1:nT
        x=trk(i).X(1); y=trk(i).X(2);
        pb=[x-trk(i).w/2,y-trk(i).h/2,x+trk(i).w/2,y+trk(i).h/2];
        for j=1:nD
            dist=hypot(x-(D(j,1)+D(j,3))/2, y-(D(j,2)+D(j,4))/2);
            v=iou(pb,D(j,1:4));
            if dist>120&&v<0.05, continue; end
            % Combined cost: distance + IoU + appearance
            distCost=min(dist/100,1);
            iouCost=1-v;
            appCost=1-appearanceSim(trk(i).hist,detHists{j});
            C(i,j)=0.35*distCost + 0.35*iouCost + 0.30*appCost;
        end
    end
    M=matchpairs(C,0.80);
    for i=1:size(M,1)
        if C(M(i,1),M(i,2))<1e5, pairs(end+1,:)=M(i,:); end %#ok<AGROW>
    end
    if ~isempty(pairs)
        freeTrk=setdiff(1:nT,pairs(:,1));
        freeDet=setdiff(1:nD,pairs(:,2));
    end
end

%% ---------- NMS ----------
function K = nms(B, thr)
    K=zeros(0,5); if isempty(B), return; end
    [~,o]=sort(B(:,5),'descend'); B=B(o,:);
    for i=1:size(B,1)
        dup=false;
        for j=1:size(K,1)
            if iou(B(i,1:4),K(j,1:4))>=thr, dup=true; break; end
        end
        if ~dup, K(end+1,:)=B(i,:); end %#ok<AGROW>
    end
end

%% ---------- IoU ----------
function v = iou(a, b)
    iw=max(0,min(a(3),b(3))-max(a(1),b(1)));
    ih=max(0,min(a(4),b(4))-max(a(2),b(2)));
    it=iw*ih; if it<=0, v=0; return; end
    v=it/((a(3)-a(1))*(a(4)-a(2))+(b(3)-b(1))*(b(4)-b(2))-it);
end

%% ---------- GT boxes from table ----------
function B = gtBoxesFromTable(T, name)
    r=strcmp(T.image,name); B=[T.x1(r) T.y1(r) T.x2(r) T.y2(r)];
end

%% ---------- Match detections to GT ----------
function [tp,fp,fn,ious] = matchBoxes(D,G,thr,minArea)
    tp=0; fp=0; ious=[];
    if isempty(G), fp=size(D,1); fn=0; return; end
    a=(G(:,3)-G(:,1)).*(G(:,4)-G(:,2));
    scored=find(a>=minArea); tiny=find(a<minArea);
    used=false(numel(scored),1);
    if ~isempty(D), [~,o]=sort(D(:,5),'descend'); D=D(o,:); end
    for i=1:size(D,1)
        best=0; bj=0;
        for j=1:numel(scored)
            if used(j), continue; end
            v=iou(D(i,1:4),G(scored(j),:));
            if v>best, best=v; bj=j; end
        end
        if bj>0&&best>=thr, tp=tp+1; used(bj)=true; ious(end+1,1)=best; continue; end %#ok<AGROW>
        onTiny=false;
        for j=1:numel(tiny)
            if iou(D(i,1:4),G(tiny(j),:))>=thr, onTiny=true; break; end
        end
        if ~onTiny, fp=fp+1; end
    end
    fn=sum(~used);
end

%% ---------- Precision / Recall / F1 ----------
function [p,r,f] = prf(tp,fp,fn)
    if tp+fp==0, p=0; else, p=tp/(tp+fp); end
    if tp+fn==0, r=0; else, r=tp/(tp+fn); end
    if p+r==0, f=0; else, f=2*p*r/(p+r); end
end

%% ---------- Kalman Predict ----------
function [X,P] = kPredict(X,P)
    F=[1 0 1 0;0 1 0 1;0 0 1 0;0 0 0 1];
    Q=4*[1/4 0 1/2 0;0 1/4 0 1/2;1/2 0 1 0;0 1/2 0 1];
    X=F*X; P=F*P*F'+Q;
end

%% ---------- Kalman Update (Joseph form) ----------
function [X,P] = kUpdate(X,P,Z)
    H=[1 0 0 0;0 1 0 0]; R=16*eye(2);
    y=Z-H*X; S=H*P*H'+R; K=P*H'/S;
    X=X+K*y; I4=eye(4);
    P=(I4-K*H)*P*(I4-K*H)'+K*R*K';
end

function [ok,tf,dxC,scale] = egoMotion(prevGray,gray,trk)
    ok=false; tf=[]; dxC=NaN; scale=NaN;
    [h,w]=size(prevGray);
    pts=detectMinEigenFeatures(prevGray,'ROI',[1 round(0.25*h) w round(0.36*h)],'MinQuality',0.01);
    if pts.Count<30, return; end
    loc=pts.selectStrongest(600).Location;
    for t=1:numel(trk)
        x=trk(t).X(1); y=trk(t).X(2);
        in=loc(:,1)>=x-trk(t).w/2&loc(:,1)<=x+trk(t).w/2&loc(:,2)>=y-trk(t).h/2&loc(:,2)<=y+trk(t).h/2;
        loc=loc(~in,:);
    end
    if size(loc,1)<30, return; end
    tracker=vision.PointTracker('MaxBidirectionalError',1.5,'BlockSize',[21 21]);
    initialize(tracker,loc,prevGray); [newLoc,valid]=tracker(gray); release(tracker);
    a_=loc(valid,:); b_=newLoc(valid,:);
    if size(a_,1)<30, return; end
    try
        tf=estgeotform2d(a_,b_,'similarity','MaxDistance',2,'Confidence',99.5,'MaxNumTrials',2000);
    catch, return; end
    c=[w/2 h/2]; cw=transformPointsForward(tf,c);
    dxC=cw(1)-c(1); scale=hypot(tf.A(1,1),tf.A(2,1)); ok=true;
end

%% ---------- Shadow fraction ----------
function frac = shadowFrac(img)
    frac=NaN; if size(img,3)~=3, return; end
    [h,w,~]=size(img); hsv=rgb2hsv(img);
    H=hsv(:,:,1)*360; S=hsv(:,:,2); V=hsv(:,:,3);
    mask=false(h,w); yT=round(0.56*h); yB=round(0.86*h);
    for y=yT:yB
        half=(0.30+(y-yT)/(yB-yT)*0.62)*w/2;
        mask(y,max(1,round(w/2-half)):min(w,round(w/2+half)))=true;
    end
    vr=V(mask); hr=H(mask); sr=S(mask);
    if isempty(vr)||median(vr)<0.20, return; end
    lit=vr>=median(vr);
    vRef=median(vr(lit)); hRef=median(hr(lit)); sRef=median(sr(lit));
    dH=abs(H-hRef); dH=min(dH,360-dH);
    sh=(V<0.62*vRef)&(dH<=12)&(S>=sRef-0.16)&mask;
    sh=imclose(imopen(sh,strel('disk',3)),strel('disk',3));
    cc=bwconncomp(sh); px=0;
    for i=1:cc.NumObjects
        if numel(cc.PixelIdxList{i})>=2000, px=px+numel(cc.PixelIdxList{i}); end
    end
    frac=px/max(sum(mask(:)),1);
end

%% ---------- Count occluded ----------
function n = countOccluded(boxes,minArea)
    n=0; if isempty(boxes), return; end
    a=(boxes(:,3)-boxes(:,1)).*(boxes(:,4)-boxes(:,2));
    b=boxes(a>=minArea,:); ab=a(a>=minArea);
    if size(b,1)<2, return; end
    cov=zeros(size(b,1),1);
    for i=1:size(b,1), for j=i+1:size(b,1)
        iw=max(0,min(b(i,3),b(j,3))-max(b(i,1),b(j,1)));
        ih=max(0,min(b(i,4),b(j,4))-max(b(i,2),b(j,2)));
        if iw*ih>0, c=iw*ih/min(ab(i),ab(j)); cov(i)=max(cov(i),c); cov(j)=max(cov(j),c); end
    end, end
    n=sum(cov>=0.30);
end

%% ---------- Draw boxes ----------
function out = drawBoxes(img,B,colour,width)
    out=img;
    for i=1:size(B,1)
        bw_=B(i,3)-B(i,1); bh_=B(i,4)-B(i,2);
        if bw_>0&&bh_>0
            out=insertShape(out,'rectangle',[B(i,1) B(i,2) bw_ bh_],'Color',colour,'LineWidth',width);
        end
    end
end

function f = letterbox(img,W,H)
    s_=min(W/size(img,2),H/size(img,1)); r=imresize(img,s_);
    f=uint8(zeros(H,W,3));
    r0=floor((H-size(r,1))/2)+1; c0=floor((W-size(r,2))/2)+1;
    f(r0:r0+size(r,1)-1,c0:c0+size(r,2)-1,:)=r;
end

%% ---------- Build KITTI sequence ----------
function buildKittiSeq(seqId,kittiBase,seqDir,maxFrames)
    imgSrc=fullfile(kittiBase,'data_tracking_image_2','training','image_02',seqId);
    labelSrc=fullfile(kittiBase,'data_tracking_label_2','training','label_02',[seqId '.txt']);
    calibSrc=fullfile(kittiBase,'data_tracking_calib','training','calib',[seqId '.txt']);
    if ~exist(imgSrc,'dir')||~exist(labelSrc,'file')
        fprintf('    WARNING: KITTI data not found for %s\n',seqId); return;
    end
    frDir=fullfile(seqDir,'frames');
    if ~exist(frDir,'dir'), mkdir(frDir); end
    srcF=dir(fullfile(imgSrc,'*.png')); nCopy=min(numel(srcF),maxFrames);
    for i=1:nCopy
        copyfile(fullfile(srcF(i).folder,srcF(i).name),fullfile(frDir,srcF(i).name));
    end
    fid=fopen(labelSrc,'r'); lines_={};
    while ~feof(fid), l=fgetl(fid); if ischar(l), lines_{end+1}=l; end, end %#ok<AGROW>
    fclose(fid);
    fout=fopen(fullfile(seqDir,'gt.csv'),'w');
    fprintf(fout,'frame,track_id,x1,y1,x2,y2,occluded,truncated,ignore\n');
    for i=1:numel(lines_)
        parts=strsplit(lines_{i});
        if numel(parts)<17, continue; end
        frame=str2double(parts{1}); if frame>=nCopy, continue; end
        tid=str2double(parts{2}); tp_=parts{3};
        tr=str2double(parts{4}); oc=str2double(parts{5});
        x1_=str2double(parts{7}); y1_=str2double(parts{8});
        x2_=str2double(parts{9}); y2_=str2double(parts{10});
        if strcmp(tp_,'Car')
            fprintf(fout,'%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%.1f,0\n',frame,tid,x1_,y1_,x2_,y2_,oc,tr);
        elseif strcmp(tp_,'DontCare')
            fprintf(fout,'%d,-1,%.2f,%.2f,%.2f,%.2f,-1,-1.0,1\n',frame,x1_,y1_,x2_,y2_);
        end
    end
    fclose(fout);
    if exist(calibSrc,'file')
        fid=fopen(calibSrc,'r');
        while ~feof(fid)
            l=fgetl(fid);
            if startsWith(l,'P2:')
                vals=sscanf(l(4:end),'%f');
                if numel(vals)>=12
                    fout=fopen(fullfile(seqDir,'calib.csv'),'w');
                    fprintf(fout,'fx,fy,cx,cy\n%.4f,%.4f,%.4f,%.4f\n',vals(1),vals(6),vals(3),vals(7));
                    fclose(fout);
                end, break;
            end
        end
        fclose(fid);
    end
    fprintf('    seq_%s: %d frames built\n',seqId,nCopy);
end

function e = edgeDensity(img) %#ok<DEFNU>
    [gx,gy]=imgradientxy(img,'sobel');
    e=mean(sqrt(double(gx).^2+double(gy).^2),'all');
end

function m = nanmean_(x)
    if isempty(x), m=NaN; return; end
    m=mean(x,'omitnan');
end

function out = drawTrackedCars(img, trk)
    out = img;
    for t = 1:numel(trk)
        x = trk(t).X(1); y = trk(t).X(2);
        w = trk(t).w; h = trk(t).h;
        bx = max(1, x - w/2); by = max(1, y - h/2);
        out = insertShape(out, 'rectangle', [bx by w h], 'Color', [0 200 255], 'LineWidth', 3);
        label = sprintf('Car #%d', trk(t).id);
        out = insertText(out, [bx max(1, by-20)], label, 'FontSize', 12, ...
            'BoxColor', [0 150 200], 'TextColor', 'white', 'BoxOpacity', 0.7);
    end
end

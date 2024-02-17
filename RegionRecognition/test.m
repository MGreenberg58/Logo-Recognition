clear;
clc;
 
%Rebecca's path
%filePath = "C:\Users\brelanre\OneDrive - Rose-Hulman Institute of Technology\Documents\MATLAB\CSSE463\Transportation";
%Nathan's path
%filePath = "C:\Users\kingnm\Documents\MATLAB\Image Recognition\Logo-Recognition\RegionRecognition\Transportation";
%Matthew's path
filePath = "E:\Logos\LogoDet-3K\Transportation";

imds = imageDatastore(filePath, "FileExtensions", [".jpg"], "IncludeSubfolders", true, "LabelSource", "foldernames");

numFiles = numel(imds.Files);
% Generate random permutation indices for shuffling
perm = randperm(numel(imds.Files));
%perm = 1:numFiles;

% Shuffle the indices and use them to access the files
shuffledFiles = imds.Files(perm);

% Calculate the number of files for each split
numTrain = round(0.6 * numFiles);
numValidation = round(0.1 * numFiles);
numTesting = numFiles - numTrain - numValidation;

bbds = fileDatastore(filePath, "ReadFcn", @readXML, "IncludeSubfolders", true, "FileExtensions", [".xml"]);

bboxes = readall(bbds);
shuffledValues = bboxes(perm); 

% Training
filename = shuffledFiles(1:numTrain);
logo = shuffledValues(1:numTrain);
trainTable = table(filename, logo);

imds = imageDatastore(trainTable.filename);
blds = boxLabelDatastore(trainTable(:,2:end));
trainds = combine(imds,blds);

%Validation
filename = shuffledFiles(numTrain+1:numTrain+numValidation);
logo = shuffledValues(numTrain+1:numTrain+numValidation);
validationTable = table(filename, logo);

imds2 = imageDatastore(validationTable.filename);
blds2 = boxLabelDatastore(validationTable(:,2:end));
validationds = combine(imds,blds);

% Testing
filename = shuffledFiles(numTrain+numValidation+1:end);
logo = shuffledValues(numTrain+numValidation+1:end);
testingTable = table(filename, logo);

imds3 = imageDatastore(testingTable.filename);
blds3 = boxLabelDatastore(testingTable(:,2:end));
testingds = combine(imds,blds);

%% Train Region Detector
inputSize = [224 224 3];
trainingDataForEstimation = transform(trainds,@(data)preprocessData(data,inputSize));

numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
};

detector = yolov4ObjectDetector("csp-darknet53-coco",{'logo'},anchorBoxes,InputSize=inputSize);

options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.0001,...
    LearnRateSchedule="none",...
    MiniBatchSize=16,...
    L2Regularization=0.0005,...
    MaxEpochs=8,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=50,...
    ValidationFrequency=500,...
    CheckpointPath=tempdir,...
    Plots="training-progress",...
    ValidationData=validationds);

trainedDetector = trainYOLOv4ObjectDetector(trainds,detector,options);

%% Display Example Image

I = imread(shuffledFiles{numTrain + 2, 1});
[bboxes, scores, labels] = detect(trainedDetector,I,Threshold=0.05);

label_str = cell(length(labels),1);
for i = 1:length(labels)
    label_str{i} = ['Confidence: ' num2str(scores(i), '%0.2f') '%'];
end

detectedImg = insertObjectAnnotation(I,"Rectangle",bboxes,label_str);
figure
imshow(detectedImg)

%% Save data
save("datector.mat")

%% Train Classifier
imds4 = imageDatastore("E:\Logos\CroppedCleaned", "IncludeSubfolders", true, "LabelSource", "foldernames");

perm = randperm(numel(imds4.Files));
shuffledFiles = imds4.Files(perm);

numTrain = round(0.7 * numel(imds4.Files));

alexTrain = imageDatastore(shuffledFiles(1:numTrain), "IncludeSubfolders", true, "LabelSource", "foldernames");
alexValidation = imageDatastore(shuffledFiles(numTrain+1:end), "IncludeSubfolders", true, "LabelSource", "foldernames");

net = resnet50;
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
[parent, ~, ~] = fileparts(imds4.Files);
[~, classes, ~] = fileparts(parent);
numClasses = numel(unique(classes));

lgraph = layerGraph(net);

lgraph = replaceLayer(lgraph, 'fc1000', fullyConnectedLayer(numClasses,'WeightLearnRateFactor', 15,'BiasLearnRateFactor', 15));
lgraph = replaceLayer(lgraph, 'fc1000_softmax', softmaxLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', classificationLayer);

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter('RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), alexTrain, 'DataAugmentation', imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), alexValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 25, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    ValidationFrequency=250, ...
    ValidationData=augimdsValidation);

netTransfer = trainNetwork(augimdsTrain, lgraph, options);

save("classifier.mat")

%% Get results

bboxResults = detect(trainedDetector, testingds);
metrics = evaluateObjectDetection(bboxResults, testingds);
classID = 1;
precision = metrics.ClassMetrics.Precision{classID};
recall = metrics.ClassMetrics.Recall{classID};

predicted = cell(numTesting, 1);
actual = cell(numTesting, 1);
countBad = 0;

numTesting = numFiles - numTrain - numValidation;

for i = 1:numTesting

    file = testingds.UnderlyingDatastores{1, 1}.Files{i, 1};
    img = imread(file);
    [boxes, scores, labels] = detect(trainedDetector, img, Threshold=0.05);

    numBoxes = numel(scores);

    [parent, ~, ~] = fileparts(file);
    [~, class, ~] = fileparts(parent);

    newPred = cell(numel(scores), 1);
    for j = 1:numel(scores)
        image = imcrop(img, boxes(j, 1:4));

        [YPred, score] = classify(netTransfer, imresize(image, [224, 224]));
        newPred(j) = cellstr(YPred);
    end

    if (numBoxes == 0)
        countBad = countBad + 1;
        continue
    end

    [uniqueStrings, ~, idx] = unique(newPred);
    modeIndex = mode(idx); %find most common
    mostFrequent = uniqueStrings(modeIndex);

    parts = strsplit(mostFrequent{1, 1}, '-');
    truncated = [parts{1}, '-'];

    predicted{i} =  truncated;

    parts = strsplit(class, '-');
    truncated = [parts{1}, '-'];
    
    actual{i} = truncated;

end

good = sum(cellfun(@(x, y) strcmp(x, y), predicted, actual));

acc = good / numel(predicted);

%% Display Results
metrics.ClassMetrics.Precision
(2 * mean(metrics.ClassMetrics.Precision{classID}) * mean(metrics.ClassMetrics.Recall{classID})) / (mean(metrics.ClassMetrics.Precision{classID}) + mean(metrics.ClassMetrics.Recall{classID}))
figure
plot(recall,precision)
xlabel("Recall")
ylabel("Precision")
grid on
title(sprintf("Average Precision = %.2f",metrics.ClassMetrics.mAP(classID)))

%% Helper Functions
function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

%for ii = 1:size(data,1)
%    I = data{ii,1};
%    imgSize = size(I);
%    
%    bboxes = data{ii,2};
%    I = im2single(imresize(I,targetSize(1:2)));
%    scale = targetSize(1:2)./imgSize(1:2);
%    bboxes = bboxresize(bboxes,scale);
    
%    data(ii,1:2) = {I,bboxes};
%end

% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
data{2} = bboxresize(boxEstimate,scale);
end

function detector = downloadPretrainedYOLOv4Detector()
% Download a pretrained yolov4 detector.
if ~exist("yolov4CSPDarknet53VehicleExample_22a.mat", "file")
    if ~exist("yolov4CSPDarknet53VehicleExample_22a.zip", "file")
        disp("Downloading pretrained detector...");
        pretrainedURL = "https://ssd.mathworks.com/supportfiles/vision/data/yolov4CSPDarknet53VehicleExample_22a.zip";
        websave("yolov4CSPDarknet53VehicleExample_22a.zip", pretrainedURL);
    end
    unzip("yolov4CSPDarknet53VehicleExample_22a.zip");
end
pretrained = load("yolov4CSPDarknet53VehicleExample_22a.mat");
detector = pretrained.detector;
end

function validateInputData(ds)
% Validates the input images, bounding boxes and labels and displays the 
% paths of invalid samples. 

% Copyright 2021 The MathWorks, Inc.

% Path to images
info = ds.UnderlyingDatastores{1}.Files;

ds = transform(ds, @isValidDetectorData);
data = readall(ds);

validImgs = [data.validImgs];
validBoxes = [data.validBoxes];
validLabels = [data.validLabels];

msg = "";

if(any(~validImgs))
    imPaths = info(~validImgs);
    str = strjoin(imPaths, '\n');
    imErrMsg = sprintf("Input images must be non-empty and have 2 or 3 dimensions. The following images are invalid:\n") + str;
    msg = (imErrMsg + newline + newline);
end

if(any(~validBoxes))
    imPaths = info(~validBoxes);
    str = strjoin(imPaths, '\n');
   boxErrMsg = sprintf("Bounding box data must be M-by-4 matrices of positive integer values. The following images have invalid bounding box data:\n") + str;
    
    msg = (msg + boxErrMsg + newline + newline);
end

if(any(~validLabels))
    imPaths = info(~validLabels);
    str = strjoin(imPaths, '\n');
    labelErrMsg = sprintf("Labels must be non-empty and categorical. The following images have invalid labels:\n") + str;
    
    msg = (msg + labelErrMsg + newline);
end

if(~isempty(msg))
    error(msg);
end

end

function out = isValidDetectorData(data)
% Checks validity of images, bounding boxes and labels
for i = 1:size(data,1)
    I = data{i,1};
    boxes = data{i,2};
    labels = data{i,3};

    imageSize = size(I);
    mSize = size(boxes, 1);

    out.validImgs(i) = iCheckImages(I);
    out.validBoxes(i) = iCheckBoxes(boxes, imageSize);
    out.validLabels(i) = iCheckLabels(labels, mSize);
end

end

function valid = iCheckImages(I)
% Validates the input images.

valid = true;
if ndims(I) == 2
    nDims = 2;
else
    nDims = 3;
end
% Define image validation parameters.
classes        = {'numeric'};
attrs          = {'nonempty', 'nonsparse', 'nonnan', 'finite', 'ndims', nDims};
try
    validateattributes(I, classes, attrs);
catch
    valid = false;
end
end

function valid = iCheckBoxes(boxes, imageSize)
% Validates the ground-truth bounding boxes to be non-empty and finite.

valid = true;
% Define bounding box validation parameters.
classes = {'numeric'};
attrs   = {'nonempty', 'integer', 'nonnan', 'finite', 'positive', 'nonzero', 'nonsparse', '2d', 'ncols', 4};
try
    validateattributes(boxes, classes, attrs);
    % Validate if bounding box in within image boundary.
    validateattributes(boxes(:,1)+boxes(:,3)-1, classes, {'<=', imageSize(2)});
    validateattributes(boxes(:,2)+boxes(:,4)-1, classes, {'<=', imageSize(1)}); 
catch
    valid = false;
end
end

function valid = iCheckLabels(labels, mSize)
% Validates the labels.

valid = true;
% Define label validation parameters.
classes = {'categorical'};
attrs   = {'nonempty', 'nonsparse', '2d', 'ncols', 1, 'nrows', mSize};
try
    validateattributes(labels, classes, attrs);
catch
    valid = false;
end
end

function box = getBBox(path)
    xmlStruct = readstruct(path);
        
    values = xmlStruct.object.bndbox;
    width = values.xmax - values.xmin;
    height = values.ymax - values.ymin;

    box = [values.xmin values.ymin width height]';
end

function data = readXML(filename) 

    xml = readstruct(filename).object.bndbox;
    xmin = xml.xmin;
    ymin = xml.ymin;
    width = xml.xmax - xml.xmin;
    height = xml.ymax - xml.ymin;

    if xmin == 0
        xmin = 1;
    end
    if ymin == 0
        ymin = 1;
    end

    data = [xmin ymin width height];

end




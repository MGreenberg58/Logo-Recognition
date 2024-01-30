clear;
clc;
 
%Rebecca's path
%filePath = "C:\Users\brelanre\OneDrive - Rose-Hulman Institute of Technology\Documents\MATLAB\CSSE463\Transportation";
%Nathan's path
%filePath = "C:\Users\kingnm\Documents\MATLAB\Image Recognition\Logo-Recognition\RegionRecognition\Transportation";
%Matthew's path
filePath = "E:\Logos\LogoDet-3K\Transportation";

ds = imageDatastore(filePath, ...
    FileExtensions=[".jpg"], ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
nImages = numel(ds.Files);
images = ds.Files;

ds1 = imageDatastore(filePath, ...
    FileExtensions=[".xml"], ...
    IncludeSubfolders=true);
xmls = ds1.Files;

imagepaths = strings(nImages,1);
bb = {nImages,1};
for i = 1:nImages
    imagepaths(i) = string(images(i));
   
    name = xmls{i};
    xml = readstruct(name).object.bndbox;
    bb(i,:) = {[xml.xmin xml.ymin xml.xmax xml.ymax]};
end 

bb = bb(:,1);

%% Generate training, validation, and test sets
rng("default");

% Get a list of subfolders in the main folder
subfolders = dir(filePath);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

trainingPercentage = 60;
validationPercentage = 10;
testingPercentage = 30;

trainingData = {};
trainingbb = {};
validationData = {};
validationbb = {};
testingData = {};
testingbb = {};

for i = 1:length(subfolders)
    subfolder = fullfile(filePath, subfolders(i).name);
    
    % Get a list of files in the current subfolder
    imgFiles = dir(fullfile(subfolder, '*.jpg'));
    xmlFiles = dir(fullfile(subfolder, '*.xml'));
    
    % Randomly shuffle the list of files
    perm = randperm(length(imgFiles));

    % Calculate the number of files for each set
    numFiles = length(imgFiles);
    numTraining = round(trainingPercentage/100 * numFiles);
    numValidation = round(validationPercentage/100 * numFiles);
    numTesting = numFiles - numTraining - numValidation;
    
    % Divide the files into training, validation, and testing sets
    
    for j = 1:numTraining
        trainingData = [trainingData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        trainingbb = [trainingbb; getBBox(fullfile(subfolder, xmlFiles(perm(j)).name))'];
    end

    for j = numTraining+1:numTraining+numValidation
        validationData = [validationData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        validationbb = [validationbb; getBBox(fullfile(subfolder, xmlFiles(perm(j)).name))'];
    end

    for j = numTraining+numValidation+1:numFiles
        testingData = [testingData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        testingbb = [testingbb; getBBox(fullfile(subfolder, xmlFiles(perm(j)).name))'];
    end

    disp(['Processing subfolder: ' subfolders(i).name]);
    disp(['  Training set: ' num2str(numTraining) ' files']);
    disp(['  Validation set: ' num2str(numValidation) ' files']);
    disp(['  Testing set: ' num2str(numTesting) ' files']);

end

%shuffledIndices = randperm(nImages);
%idx = floor(0.6 * length(shuffledIndices) );

%trainingIdx = 1:idx;
%trainingData = imagepaths(shuffledIndices(trainingIdx),:);
%trainingbb = table(bb(shuffledIndices(trainingIdx),:));

%validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
%validationData = imagepaths(shuffledIndices(validationIdx),:);
%validationbb = table(bb(shuffledIndices(validationIdx),:));

%testIdx = validationIdx(end)+1 : length(shuffledIndices);
%testData = imagepaths(shuffledIndices(testIdx),:);
%testbb = table(bb(shuffledIndices(testIdx),:));

%% Saving off data
save('RegionData.mat');

%% Load Data
load("RegionData.mat")

%% Generate datastores
imgTrain = imageDatastore(trainingData);
boxTrain = boxLabelDatastore(table(trainingbb));

imgValidation = imageDatastore(validationData);
boxValidation = boxLabelDatastore(table(validationbb));

imgTest = imageDatastore(testingData);
boxTest = boxLabelDatastore(table(testingbb));

trainingData = combine(imgTrain,boxTrain);
validationData = combine(imgValidation,boxValidation);
testData = combine(imgTest,boxTest);

%% Validate training set
validateInputData(trainingData);
validateInputData(validationData);
validateInputData(testData);

%% Display example image
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Prep network and Augment Data
%Input needs to bea multiple of 32, so 224 instead of 227
inputSize = [224 224 3];
className = "logo";
rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };
detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=100,...
    L2Regularization=0.0005,...
    MaxEpochs=2,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=1000,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

%% Train the Network

%% Creates helper functions from example online
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
    boxErrMsg = sprintf("Bounding box data must be M-by-4 matrices of positive integer values. The following images have invalid bounding box data:\n") ...
        + str;
    
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
    box = [values.xmin values.ymin values.xmax values.ymax]';
end












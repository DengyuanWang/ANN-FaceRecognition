classdef HandWriting_recognition
    properties(Access = private)
        DataSet_path  %folder name of dataset(the folder need to be at the same folder of class.m)
        Classifers
        Categories_nums%nums of categories
        Total_dataset% every row is [X Y];
        Train_dataset% every row is [X Y];
        Test_dataset%  every row is [X Y];
        Regular_img_size = [100 100];
        Resize_tag = true;
        Img_mean=[];
        coeff_forpca=[];
        NeuralNet;
        Trained_net;
    end
    methods(Access=public)
        function obj =  HandWriting_recognition()%load trained svm tag
                obj.Train_dataset = [];
                obj.Test_dataset = [];
                [obj,Categories_nums] = obj.load_dataset('English/Hnd');
                obj.Categories_nums = Categories_nums;
                tmp = obj.Total_dataset(1,:);
                X = tmp{1,1};Y = tmp{1,2};
                InputLayer = size(X,2);
                HiddenLayers = [100  100 100]';
                OutputLayer =  size(Y,2);
                LearningRate = 0.01;
                obj.NeuralNet = ANN(InputLayer, HiddenLayers, OutputLayer,LearningRate);
        end
        function [obj, Categories_nums] = load_dataset(obj,foldername)
            if(exist(foldername,'dir')==7)
                Datas_tmp = [];
                addpath(foldername);
                if(exist('all.txt','file')~=2)
                   fprintf("error, all.txt not exist\n");
                else
                    Paths = textread(strcat(foldername,'/all.txt'),'%s');
                    Paths = cellfun(@(x) foldername+"/"+x,Paths);
                end
                dataset = [];
                for i=1:size(Paths,1)
                    path = Paths(i);
                    X = imread(path);
                    X = rgb2gray(X);
                    if(obj.Resize_tag)
                        X = imresize(X,obj.Regular_img_size);
                    end
                    %resize X into a row vector
                    X = double(reshape(X,1,size(X,1)*size(X,2)))./255;
                    %get category index Y
                    expression = 'Sample\d*';
                    matchStr = regexp(path,expression,'match');
                    Y = str2num(extractAfter(matchStr,'Sample'));
                    dataset = [dataset; [{X} {Y}]];
                end
                Y = dataset(:,2);
                Y = cell2mat(Y);
                Categories_nums = max(Y);
                Y = arrayfun(@(x) in(x,Categories_nums),Y,'Uniformoutput',false);
                dataset(:,2) = Y;
                obj.Total_dataset = dataset;
            end
            function y = in(y,Categories_nums)
                t = zeros(1,Categories_nums);
                t(y) = 1;
                y = t;
            end
        end
        function obj = fix(obj)%train nerual network
            %callculate classifier
            net1 = obj.NeuralNet;
            batch_size = 32;
            Batches = get_batches(obj.Train_dataset,batch_size);

            for i=1:length(Batches)
                fprintf('rate:%d/%d\n',i,length(Batches));
                delta = repmat({0},batch_size,1);
                batch = Batches{i};
                parfor j=1:batch_size
                    tmp = batch{j}
                    t = net1.backpropagation(tmp{1},tmp{2});
                    delta{j} = t.get_deltatheta();
                end
                sample = delta{1};
                result = repmat({0},size(sample));
                for j=1:length(delta)
                    sample = delta{j};
                    for k=1:length(sample)
                        result{k} = result{k}+ sample{k}./batch_size;
                    end
                end
                net1 = net1.set_theta(result);
            end
            obj.Trained_net = net1;
            function Batches = get_batches(dataset,batchz_size)
                len = ceil(size(dataset,1)/batchz_size);
                lenmax = size(dataset,1);
                Batches = repmat({0},len,1);
                parfor i1=1:len
                    startindex = batchz_size*(i1-1)+1;
                    endindex = min(batchz_size*(i1),lenmax);
                    Batches{i1} = dataset(startindex:endindex,:);
                end
            end
%             
%             for i=1:size(obj.Train_dataset,1)
%                 tmp = obj.Train_dataset{i};
%                 X = tmp{1,1};Y = tmp{1,2};
%                 obj.Trained_net = obj.Trained_net.backpropagation(X,Y);
%                 fprintf('processing\n');
%             end
%             fprintf("Train over\n");
        end
        function accuracy = Five_fold_Cross_validation(obj)
            Dataset= mat2cell(obj.Total_dataset,ones(1,size(obj.Total_dataset,1)));
            index = (1:1:size(obj.Total_dataset,1))';
            %shuffle the index randomly
            index=index(randperm(length(index)));
            groupsize = size(index,1)/5;
            accuracy = [];
            %% calculate all five gourps of indexes
            for i=1:5
                index_test = index( (i-1)*groupsize+1 : min(i*groupsize,size(index,1)) );
                index_train = index;
                index_train( (i-1)*groupsize+1 : min(i*groupsize,size(index,1)) ) = [];
                Train = Dataset(index_train);
                Test = Dataset(index_test);
                obj.Train_dataset = Train;
                obj.Test_dataset =  Test;
                accuracy = [accuracy obj.validate_on_current_data()];
                fprintf('%d/n',i*20);
            end
            accuracy = mean(accuracy);
        end
        function accuracy = validate_on_current_data(obj)
            obj = obj.fix();
            Testdataset = obj.Test_dataset;
            c_net = obj.Trained_net;
            result = zeros(size(Testdataset,1),1);
            parfor i=1:size(Testdataset,1)
                tmp = obj.Testdataset{i};
                X = tmp{1,1};Y_stander= tmp{1,2};
                Y_out = c_net.predixt(X);
                [~,I1]=max(Y_stander);
                [~,I2]=max(Y_out);
                result(i)= I1==I2;
            end
            accuracy = sum(result,1)/size(result,1);
        end
        function classindex = predict(obj,point)
            Diff = sum((obj.Classifers(:,1:end-1) - repmat(point,size(obj.Classifers(:,1:end-1),1),1)).^2,2);
            [~, I] = sort(Diff);
            classindex = obj.Classifers(I(1),end);
        end
    end
end
            
    
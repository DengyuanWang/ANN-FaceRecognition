classdef HandWriting_recognition
    properties(Access = private)
        DataSet_path  %folder name of dataset(the folder need to be at the same folder of class.m)
        Classifers
        Categories_nums%nums of categories
        Total_dataset% every row is [X Y];
        Train_dataset% every row is [X Y];
        Test_dataset%  every row is [X Y];
        Regular_img_size = [35 35];
        Resize_tag = true;
        Img_mean=[];
        coeff_forpca=[];
        NeuralNet;
        Trained_net;
        LearningRate
    end
    methods(Access=public)
        function obj =  HandWriting_recognition()%load trained svm tag
                obj.Train_dataset = [];
                obj.Test_dataset = [];
                [obj,Categories_nums] = obj.load_dataset('English/Hnd');
                obj.Categories_nums = Categories_nums;
                tmp = obj.Total_dataset(1,:);
                X = tmp{1,1};Y = tmp{1,2};
                sizes = [{0} {100} {62}];
                sizes{1} = length(X);
                sizes{end} = length(Y);
                obj.LearningRate = 1.5;
                obj.NeuralNet = ANN_imm(sizes);
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
                        X = 2*(double(X)./255-0.5);
                    end
                    %resize X into a row vector
                    X = double(reshape(X,1,size(X,1)*size(X,2)));
                    %get category index Y
                    expression = 'Sample\d*';
                    matchStr = regexp(path,expression,'match');
                    Y = str2num(extractAfter(matchStr,'Sample'));
                    dataset = [dataset; repmat( [{X} {Y}],2,1)];
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
        end
        function accuracy = validation(obj)
            Dataset= mat2cell(obj.Total_dataset,ones(1,size(obj.Total_dataset,1)));
            index = (1:1:size(obj.Total_dataset,1))';
            %shuffle the index randomly
            index=index(randperm(length(index)));
            groupsize = size(index,1)/20;
            Test_data =  Dataset(index(1:groupsize));
            Train_data = Dataset(index(1+groupsize:end));
            accuracy = [];
            epochs = 10000;
            mini_batch_size = 10;
            obj.NeuralNet = obj.NeuralNet.SGD(Train_data,epochs,mini_batch_size,obj.LearningRate,Test_data);
        end
    end
end
            
    
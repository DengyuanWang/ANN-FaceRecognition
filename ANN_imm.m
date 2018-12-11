classdef ANN_imm
   properties
       num_layers 
       sizes
       biases
       weights
       nabla_b_zero
       nabla_w_zero 
   end
    methods
        function obj = ANN_imm(sizes)
            %sizes is a cell array store size from input to output
            %the bias is not included
            obj.num_layers = length(sizes);
            obj.sizes = sizes;
            rng('default');% For reproducibility
            %no bias for input layer
            obj.biases = cellfun(@(y) normrnd(0,1,[y,1]),sizes(2:end),'Uniformoutput',false);
            %weight :[currenL,nextL]
            obj.weights = cellfun(@(x,y) normrnd(0,1,[y,x]),sizes(1:end-1),sizes(2:end),'Uniformoutput',false);
            obj.nabla_b_zero= cellfun(@(y) zeros([y,1]),obj.sizes(2:end),'Uniformoutput',false);
            obj.nabla_w_zero = cellfun(@(x,y) zeros([y,x]),obj.sizes(1:end-1),obj.sizes(2:end),'Uniformoutput',false);
        end
        function a_out =feedforward(obj,a_in)
            a = a_in';%currentNodes*1
            for i=1:length(obj.biases)
                b = obj.biases{i};%size = nextNodes*1
                w = obj.weights{i};%size = nextNodes*currentNodes
                a = sigmoid(w*a+b);%size = nextNodes*1
            end
            a_out = a;
        end
        function obj = SGD(obj,training_data,epochs,mini_batch_size,eta,test_data)
            %% Train current network using mini-batch stochastic gradient descent
            % training_data is cell list format as [{{X} {Y}} ...]
            % if test_data are given, then we will evaluate after every
            % epoch
            test_num = 0;
            if nargin>4%more than 4 argument = test data are given
                test_num = length(test_data);
            end
            train_num = length(training_data);
            for j=1:epochs
                %shuffle the index randomly
                training_data = training_data( randperm(length(training_data)) );
                parfor i=1:ceil(train_num/mini_batch_size)
                    spos = 1+(i-1)*mini_batch_size;
                    epos = min(i*mini_batch_size, train_num)
                    mini_batches{i} = training_data(spos:epos);
                end
                for i=1:length(mini_batches)
                    obj = obj.update_mini_batch(mini_batches{i},eta);
                end
                if test_num>0
                    fprintf("Epoch%d:%d / %d\n",j,obj.evaluate(test_data),test_num);
                end
            end
        end
        
        function obj = update_mini_batch(obj, mini_batch, eta)
            %% Update weights and bias of the network
                %    Update weights and bias of the network using backpropagation
                %    and gradient descent on a mini batch
                %    mini_batch is a cell list  [{ {X} {Y} }....]; 
                %    eta is the learning rate
                len_batch = length(mini_batch);
                tmp = repmat({0},1,len_batch);
                nabla_b = obj.nabla_b_zero;
                nabla_w = obj.nabla_w_zero;
                for i=1:len_batch
                    turple = mini_batch{i};
                    [delta_nabla_b,delta_nabla_w] = obj.backprop(turple{1},turple{2});
                    for j=1:(obj.num_layers-1)
                        nabla_b{j} = nabla_b{j} + delta_nabla_b{j};
                        nabla_w{j} = nabla_w{j} + delta_nabla_w{j};
                    end
                end
                for j=1:(obj.num_layers-1)
                    obj.weights{j} = obj.weights{j} - (eta/len_batch)*nabla_w{j};
                    obj.biases{j} = obj.biases{j} - (eta/len_batch)*nabla_b{j};
                end
        end
        function [delta_nabla_b,delta_nabla_w] = backprop(obj,X,Y)
            %% calculate gradient of cost function for bias and weight
            %delta_nabla_b is a cell list
            %delta_nabla_w is a cell list
            X = X';
            Y = Y';
            nabla_b = obj.nabla_b_zero;
            nabla_w = obj.nabla_w_zero;
          %% feedforward
            activation = X;
            all_activations ={X};
            zs  = [];
            for i=1:length(obj.biases)
                b = obj.biases{i};w = obj.weights{i};
                z = w*activation + b;
                zs = [zs {z}];
                activation = sigmoid(z);
                all_activations = [all_activations {activation}];
            end
            %% backward pass
            delta = obj.cost_derivative(all_activations{end},Y).*sigmoid_prime(zs{end});
            nabla_b{end} = delta;
            nabla_w{end} = delta*all_activations{end-1}';
            %% l =1 means the last layer of neurons
            for L=2:obj.num_layers-1
                z = zs{end+1-L};
                sp = sigmoid_prime(z);
                delta = obj.weights{end+1-L+1}'*delta.*sp;
                nabla_b{end+1-L} = delta;
                tmp = all_activations{end+1-L-1}';
                nabla_w{end+1-L} = delta*tmp;
            end
            delta_nabla_b = nabla_b;
            delta_nabla_w = nabla_w;
        end
        function accuracy = evaluate(obj, test_data)
            %% return number of correct predict
            % note that the highest value output neruon index is class
            % index
            test_result = zeros(size(test_data));
            test_dis = zeros(size(test_data));
            for i=1:length(test_data)
                turple = test_data{i};
                Y = turple{2};Y = Y';
                pdx = obj.feedforward(turple{1});
                [~,I1] = max(pdx);
                [~,I2] = max(Y);
                 test_dis(i) = sqrt(sum((pdx-Y).^2));
                test_result(i) = I1==I2;
            end
            fprintf('dis:%f\n', sum(test_dis)/length(test_data));
            accuracy = sum(test_result);
        end
        function derevative = cost_derivative(obj, output_activations, y)
            derevative = output_activations - y;
        end
    end
end
function z = sigmoid(z)
     z = 1./(1+exp(-z));
end
function result = sigmoid_prime(z)
    result = sigmoid(z).*(1-sigmoid(z));
end
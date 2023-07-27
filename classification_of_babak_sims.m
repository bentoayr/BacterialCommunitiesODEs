% load data
load 'all_graph_n_8_all_1_perturbations_11pm_28th_june_2021.mat';

%all_results = cell(2^10 - 1,n_max,4);
%all_results{104}{1}{1} = atoms
%all_results{104}{1}{2} = distribution of atoms
%all_results{104}{1}{3} = distribution of first, second and third y values
%all_results{104}{1}{4} = starting support for first stable community







% acho que o problema e' que devia quere mostrar que consigo fazer
% previsoes que sao de certa forma universais. Ou seja, que com a mesma
% caixa, consigo fazer previsoes para qualquer rede. No entanto, para isso
% tinha que fazer simulac,oes usando diversos parametros para a minha rede,
% o que actualmente nao estou a fazer




% tenho que construir um classificador que dado   
% start_stable_state_community, edges for stable_state_community , invador added  0 0 0 0 0 1 0 0 0   ----> me preve se o invasor vai ficar ou nao       
% start_stable_state_y ,edges for stable_state_community, invador added  0 0 0 0 0 y_value 0 0 0   ----> me preve se o invasor vai ficar ou nao       
% start_stable_state_ratios ,edges for stable_state_community, invador added  0 0 0 0 0 ratio 0 0 0   ----> me preve se o invasor vai ficar ou nao   

% start_stable_state_community ,start_stable_state_ratios , start_stable_state_y ,  edges_for_stable_state_community, invador_added  0 0 0 0 0 ratio 0 0 0   ----> me preve se o invasor vai ficar ou nao   


%% first start with a compressed data. such that we do not need to expand things too much
compressed_configurations = [];
compressed_probs = [];
states = [];

num_reps = 1000;

for base_comm_ix = 1 : 2^n_max - 1
    for invader_r_ix = 1:n_max
        
        [atoms_of_dist_first_and_second_stable_set,prob_evol] = get_atoms_and_dist_from_full_dist(all_results,base_comm_ix,invader_r_ix,n_max,eps_tol,num_reps);
        
       
        
        dist_size = size(atoms_of_dist_first_and_second_stable_set,1);
        config_evol = [repmat(all_results{base_comm_ix}{invader_r_ix}{4}',dist_size ,1),zeros(dist_size,n_max), atoms_of_dist_first_and_second_stable_set];
        config_evol(:,n_max + invader_r_ix) = 1;
        prob_evol = prob_evol';
       
     
        % remove all configuations where the perburbation is not adding an invader
        ix_where_addition_will_not_happen = atoms_of_dist_first_and_second_stable_set(:,invader_r_ix) == 1;
        config_evol(ix_where_addition_will_not_happen,:) = [];
        prob_evol(ix_where_addition_will_not_happen) = [];
  
        compressed_configurations = [compressed_configurations; config_evol(:,1:end-n_max)]; % we do not add the final state, otherwise the problem is trivial
        compressed_probs = [compressed_probs ; prob_evol];

        states = [states; config_evol(:,end-n_max+invader_r_ix)];
        
    end
end

%% resample tomake things more manageable and also accurate
idx_for_sample = (cumsum(compressed_probs)/sum(compressed_probs));
num_samples = 20000;
resampledfeatures = nan(num_samples, size(compressed_configurations,2));
resampledlabels = nan(num_samples,1);
for s_ix = 1:num_samples

    new_ix = 1+sum(idx_for_sample < rand);
    resampledfeatures(s_ix,:) = compressed_configurations(new_ix,:);
    resampledlabels(s_ix) = states(new_ix);
    
end

testfeatures = resampledfeatures(end-(num_samples/2) + 1:end,:);
testlabels = resampledlabels(end-(num_samples/2) + 1:end);

resampledfeatures = resampledfeatures(1:(num_samples/2),:);
resampledlabels = resampledlabels(1:(num_samples/2));



Mdl_tree = fitctree((resampledfeatures),resampledlabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Holdout',0.3));

Mdl_naive_bayes = fitcnb((resampledfeatures),resampledlabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Holdout',0.3));

Mdl_svm = fitclinear((resampledfeatures),resampledlabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Holdout',0.3,'MaxObjectiveEvaluations',100));

Mdl_knn = fitcknn((resampledfeatures),resampledlabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Holdout',0.3));

Mdl_auto = fitcauto((resampledfeatures),resampledlabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Holdout',0.3));





figure;
for class_alg = 1:5
    switch class_alg
        case 1
            Mdl = Mdl_tree;
        case 2
            Mdl = Mdl_naive_bayes;
        case 3
            Mdl = Mdl_auto;  
        case 4
            Mdl = Mdl_knn;
        case 5
            Mdl = Mdl_svm;
            
    end
    
    [label,score] =  Mdl.predict(testfeatures);
    [X,Y] = perfcurve(testlabels,score(:,2),1);
    hold on;
    plot(X,Y);
end
plot(0:0.1:1,0:0.1:1)
legend('decision tree','naive bayes','auto','knn','svm','random')
xlabel('false positive rate')
ylabel('true positive rate')




%Mdl = fitcknn((resampledfeatures),resampledlabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Holdout',0.3));

%svmmodel = fitcsvm(resampledfeatures,resampledlabels,'OptimizeHyperparameters','auto');

C_Model = crossval(Mdl);

classLoss = kfoldLoss(C_Model);

disp([sum(resampledlabels)/num_samples , classLoss]);

%%



% tenho que calcular a probabilidade de saltar de qualuer tipo de estado
% estavel para qualquer tipo de estado estavel. A questao e': qual e' o invasor?


% tenho que construir uma distribuic,ao para a adic,ao de cada tipo de
% invasor. Quando adiciono invasor 1, o que pode acontecer relativamente a
% mudanca. Quando adiciono invasor 2, o que pode acontecer relativamente a
% mudanca. etc.


% para cada invasor r quero obter uma matrix T^r,i,j) por forma T(r,i,j) me
% diz a probabilidade de ir da configurac,ao i para a configuracao j quando adiciono r
% se fizer isto, entao consigo perguntar  P(      o r fica        |     r   ,    conf inicial   )
% isto pode ser um classificador muito simples
% tambe;m posso tentar calcular P(      o r fica        |     r    ) para outro classificador simples
% tambe;m posso tentar calcular P(      o r fica       ) para outro classificador ainda mais simples



% class 1
% 80%  20% 
% [p = always on] get it right 80% of the time
% [p = alpha percentage of times] get it right    p*0.8 + (1-p)*0.2 = 0.2 + 0.6*p

% class 2
% 80% 20% for r = 1 ---> 80% success
% 20% 60% for r = 2 ---> 60% success
% etc.
% class 2.1
% sera' que um classificador linear e' suficiente para codificar este tipo de problema ?


% class 3
% 80% 20% for r = 1 conf inicial = [0 1 0 0 1 0 1]   -> 80% success
% 20% 60% for r = 2 conf inicial = [0 0 1 0 1 0 1]   -> 60% success
% class 3.1
% sera' que um classificador linear e' suficiente para codificar este tipo de problema ?


% class 4
% P(      o r fica        |     r   ,    conf inicial  ,  start_stable_state_ratios  ,   start_stable_state_y,  edges )
% nao consigo resolver este problema de forma exaustiva porque nao consigo descretizar os estados intermedios
% mas em principio o classificador existe
% class 4.1
% sera' que um classificador linear e' suficiente para codificar este tipo de problema ?



% acho que o seguinte vai acontencer           class 1 < class 2.1 < class 2 < class 3.1 < class 3 <  class 4.1   < class 4


    
    
    
function [atoms_of_dist_first_and_second_stable_set,b] = get_atoms_and_dist_from_full_dist(all_results,base_comm_ix,invader_r_ix,n_max,eps_tol,num_reps)



dist_first_state = max(0,all_results{base_comm_ix}{invader_r_ix}{3}(num_reps + (1:num_reps),1:n_max));
dist_second_state = max(0,all_results{base_comm_ix}{invader_r_ix}{3}(2*num_reps + (1:num_reps),1:n_max));

ix_non_zero_first_state = sum(dist_first_state,2) > eps_tol;
ix_non_zero_second_state = sum(dist_second_state,2) > eps_tol;

dist_first_stable_set = 0*dist_first_state;
dist_first_stable_set(ix_non_zero_first_state,:) = (dist_first_state(ix_non_zero_first_state,:)./sum(dist_first_state(ix_non_zero_first_state,:),2)) > eps_tol;

dist_stable_set_after_pert = 0*dist_second_state;
dist_stable_set_after_pert(ix_non_zero_second_state,:) = (dist_second_state(ix_non_zero_second_state,:)./sum(dist_second_state(ix_non_zero_second_state,:),2)) > eps_tol;

[atoms_of_dist_first_and_second_stable_set,~,b] = unique([dist_first_stable_set , dist_stable_set_after_pert],'rows');
b = histcounts(b,0.5 + (0:size(atoms_of_dist_first_and_second_stable_set,1)));
b = b/sum(b);

end






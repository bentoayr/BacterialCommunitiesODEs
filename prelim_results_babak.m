%% preliminary results grant with babak

n_max = 8;
m_max = 20;
p_m = 0.4;

library = (rand(n_max,m_max) < p_m);
S_vecs = randn(m_max,n_max,3);
S_vecs(:,:,1) = S_vecs(:,:,1).*(library'); % this is r_ij in model 3
S_vecs(:,:,2) = S_vecs(:,:,2).*(library');  % this is alpha_ij - beta_ij in model 3

S_vecs(:,:,3) = repmat( randn(m_max,1)  , 1,n_max,1); % this is K_j in model 3
S_vecs(:,:,3) = abs(S_vecs(:,:,3));

p_n = 0.8;

delta = 0.005;

eps_tol = 10^(-4);

%%
choice_n = rand(n_max,1) < p_n;

%% go over all possible graphs

all_results = cell(2^n_max - 1,n_max,4);
list_of_empty_stuff = 1 : 2^n_max - 1;

%delete(gcp('noCreate'));
%parpool(20);
for choice_n_index = 1:2^n_max -1
    %choice_n_index = list_of_empty_stuff(choice_n_index_ix);
    
    if (ismember(choice_n_index,list_of_empty_stuff))
        
        
        
        b = dec2bin(choice_n_index);
        g = bin2gray(b);
        choice_n = de2bi(bin2dec(g),n_max)'==1;
        
        %%
        
        for pert_ix = 1:n_max
            
            tic();
            max_num_reps = 1000;
            
            dist_start_state = nan(max_num_reps , n_max+m_max);
            dist_first_state = nan(max_num_reps , n_max+m_max);
            dist_second_state = nan(max_num_reps , n_max+m_max);
            dist_first_stable_set = nan(max_num_reps , n_max);
            dist_stable_set_after_pert = nan(max_num_reps , n_max);
            
            for rep = 1:max_num_reps
                
                [tmp1, tmp2,tmp3,tmp4, tmp5 ] = compute_distributions(choice_n,library,delta,S_vecs,eps_tol,pert_ix);
                
                dist_start_state(rep,:) = tmp1;
                dist_first_state(rep,:) = tmp2;
                dist_second_state(rep,:) = tmp3;
                dist_first_stable_set(rep,:) = tmp4;
                dist_stable_set_after_pert(rep,:) = tmp5;
                
            end
            
            
            [atoms_of_dist_first_and_second_stable_set,~,b] = unique([dist_first_stable_set , dist_stable_set_after_pert],'rows');
            %b = histcounts(b);
            b = histcounts(b,0.5 + (0:size(atoms_of_dist_first_and_second_stable_set,1)));

            b = b/sum(b);
            
            time_per_graph = toc();
            
            disp([ choice_n_index , pert_ix, time_per_graph]);
            
            all_results{choice_n_index}{pert_ix}{1} = atoms_of_dist_first_and_second_stable_set;
            all_results{choice_n_index}{pert_ix}{2} = b;
            all_results{choice_n_index}{pert_ix}{3} = [dist_start_state ; dist_first_state ; dist_second_state];
            all_results{choice_n_index}{pert_ix}{4} = choice_n;
            
        end
        
    end
end


%%

function [dist_start_state, dist_first_state,dist_second_state,dist_first_stable_set, dist_stable_set_after_pert ] = compute_distributions(choice_n,library,delta,S_vecs,eps_tol,pert_ix)

n_max = size(library,1);
m_max = size(library,2);

dist_start_state = nan(1, n_max+m_max);
dist_first_state = nan(1 , n_max+m_max);
dist_second_state = nan(1 , n_max+m_max);
dist_first_stable_set = nan(1 , n_max);
dist_stable_set_after_pert = nan(1 , n_max);



S0 = choice_n.*rand(n_max,1);
C0 = (sum(library(choice_n,:)) > 0)'.*rand(m_max,1);

y0 = [S0 ; C0];

dist_start_state(1,:) = y0;

odefun = @(t,y) model3(t , y , S_vecs , choice_n ,delta );

tspan = [0, 10];
while(1)
    ode_options = odeset('NonNegative',1);
    [~,y] = ode45(odefun,tspan,y0,ode_options);
    
    ratiosy = y(:,1:n_max)./sum(y(:,1:n_max),2); % only care about the ratios of the species and not of the chemicals.
    diffratiosy = abs(diff(ratiosy,1));
    
    err = movmean(max(diffratiosy,[],2),10);
    
    if (err(end) < eps_tol || max(tspan) > 200 || sum(y(end,1:n_max)) < eps_tol)
        break;
    else
        tspan = tspan + 10;
        y0 = y(end,:);
    end
end

if (min(y(:))< -eps_tol)
    disp(['1nd round: there is an error. large negative quantities', num2str(min(y(:)))]);
end

% fix things, even if there were errors
y(:) = max(y(:),0);

% we only compute the ratios if there is anything there. If
% there is nothing there we just ignore everything
if (sum(y(end,1:n_max)) > eps_tol)
    stable_comm = ratiosy(end,1:n_max) > eps_tol;
else
    stable_comm = (zeros(1,n_max) == 1);  % all false
end

if ( min(choice_n(stable_comm)) == 0  )
    disp(['1st round: there is an error. stable community includes species that never existed. spontanous generation is not possible',num2str(y(1:n_max)),num2str(choice_n' + 0.0)]);
end

% fix things, even if there were errors
y(end,choice_n==0) = 0;
if (sum(y(end,1:n_max)) > eps_tol)
    stable_comm = ratiosy(end,1:n_max) > eps_tol;
else
    stable_comm = (zeros(1,n_max) == 1);  % all false
end

dist_first_stable_set(1,:) = stable_comm;
dist_first_state(1,:) = y(end,:);

% perturb and run again. Need to fix this code according to the code above
y0_second_round = max(y(end,:),0);
choice_n_second_round = stable_comm';

if (stable_comm(pert_ix)==1)
    y0_second_round(pert_ix) = 0;  % if it was there, remove it.
    choice_n_second_round(pert_ix) = 0;
else
    y0_second_round(pert_ix) = mean(y0_second_round(1:n_max)); % if it was not there, add it in amount equal to the average of the species. We never touch the chemicals
    choice_n_second_round(pert_ix) = 1;
end

odefun = @(t,y) model3(t , y , S_vecs , choice_n_second_round ,delta );

tspan = [0, 10];
while(1)
    
    ode_options = odeset('NonNegative',1);
    [~,y] = ode45(odefun,tspan,y0_second_round,ode_options);
    
    ratiosy = y(:,1:n_max)./sum(y(:,1:n_max),2); % only care about the ratios of the species and not of the chemicals.
    diffratiosy = abs(diff(ratiosy,1));
    
    err = movmean(max(diffratiosy,[],2),10);
    
    if (err(end) < eps_tol || max(tspan) > 100 || sum(y(end,1:n_max)) < eps_tol)
        break;
    else
        tspan = tspan + 10;
        y0_second_round = y(end,:);
    end
end

if (min(y(:))< -eps_tol)
    disp(['2nd round: there is an error. large negative quantities',num2str(min(y(:)))]);
end

y(:) = max(y(:),0); % no large negative quantities, just fix numerics

if (sum(y(end,1:n_max)) > eps_tol)
    stable_comm = ratiosy(end,1:n_max) > eps_tol;
else
    stable_comm = zeros(1,n_max)==1;
end

if ( min(choice_n_second_round(stable_comm)) == 0  )
    disp(['2nd round: there is an error. stable community includes species that never existed. spontanous generation is not possible',num2str(y(1:n_max)),num2str(choice_n_second_round'+0.0)]);
end

% fix things, even if there were errors
y(end,choice_n_second_round==0) = 0;
if (sum(y(end,1:n_max)) > eps_tol)
    stable_comm = ratiosy(end,1:n_max) > eps_tol;
else
    stable_comm = (zeros(1,n_max) == 1);  % all false
end

dist_stable_set_after_pert(1,:) = stable_comm;
dist_second_state(1,:) = y(end,:);



end



function dydt = model3( ~ , y , S_vecs , choice_n ,delta )

m_max = size(S_vecs,1);
n_max = size(S_vecs,2);

dydt = zeros(n_max+m_max,1);

S = y(1:n_max);
C = y(n_max+1:end);
K = S_vecs(:,1,3);
AmB = S_vecs(:,:,2).*(choice_n');
R = (S_vecs(:,:,1)').*choice_n;

dydt(1:n_max) = (-delta + R*(C./(C + K))).*S;

dydt(n_max+1:end) = -delta*C + (AmB*S);


tmp = dydt;
tmp( y <= 0 & tmp <= 0) = 0;
dydt = tmp;


end
%%

function g = bin2gray(b) % https://www.matrixlab-examples.com/gray-code.html
g = b;
g(1) = b(1);
for i = 2 : length(b)
    x = xor(str2double(b(i-1)), str2double(b(i)));
    g(i) = num2str(x);
end
end
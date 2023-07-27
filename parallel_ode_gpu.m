%% solve multiple ODEs in parallel on GPU using sparse matrices


%% simple test to compare the speed of GPU and CPU on sparse matrix operations
n = 1000;
A = rand(n) > 0.8;
A = sparse(A +0.0);

tic()
gpuA = gpuArray(A);
gpuA = gpuA*gpuA*gpuA;
AAA = gather(gpuA);
toc()

tic()
A*A*A;
toc()

norm(full(AAA-A*A*A))
%% trying to play with pagefun on the GPU
num_reps = 1000;
n = 10;
m = 20;
A = rand(n,m,num_reps);
v = rand(m,num_reps);

gpuA = gpuArray(A);

tic();
repv = reshape((v),m,1,num_reps);
gpuout = pagefun(@mtimes,gpuA,gpuArray(repv));
%cpugpuout = gather(gpuout);
%cpugpuout = reshape(cpugpuout,n,num_reps);
t1 = toc();

tic();
out = cpumanymult(A,v);
t2 = toc();

t2/t1

%% trying to play with pagefun on the GPU with auto repetition and inline product
num_reps = 1000;
n = 10;
m = 20;
A = rand(n,m);
v = rand(m,num_reps);

gpuA = gpuArray(A);
gpuArep = repmat(gpuA,1,1,num_reps);

%gpufun = @(x,y) x.*y';

tic();
vreshape = reshape((v),1,m,num_reps);
gpuv = gpuArray(vreshape);
gpuvrep = repmat(gpuv,n,1,1);
gpuout = arrayfun(@times,gpuArep,gpuvrep);
cpugpuout = gather(gpuout);
t1 = toc();

tic();
out = cpumanylinemult(A,v);
t2 = toc();

norm(out(:)-cpugpuout(:))

t2/t1




%% tensor form for multiple solving

num_reps = 1000;

m_max = size(S_vecs,1);
n_max = size(S_vecs,2);

% multiple input data
choice_n_rep = (rand(n_max,num_reps) > 0.5);
y_reps = rand(n_max + m_max, num_reps);
S_vecs_1_rep = repmat(S_vecs(:,:,1)',1,1,num_reps);
S_vecs_2_rep = repmat(S_vecs(:,:,2),1,1,num_reps);


%% GPU vectorized

gpu_choice_n_rep = gpuArray(choice_n_rep);
gpu_S_vecs_1_rep = gpuArray(S_vecs_1_rep);
gpu_S_vecs_2_rep = gpuArray(S_vecs_2_rep);
gpu_S_vecs_3 = gpuArray(S_vecs(:,1,3));
gpu_delta = gpuArray(delta);

gpu_choice_n_rep_aug = reshape(gpu_choice_n_rep,1,n_max,num_reps);
gpu_choice_n_reps_trans = permute(gpu_choice_n_rep_aug,[2,1,3]);
gpu_choice_n_bar_rep = repmat(  gpu_choice_n_rep_aug , m_max, 1, 1  );
gpu_choice_n_bar_rep_trans = repmat(  gpu_choice_n_reps_trans , 1, m_max, 1  );

%gpu_AmB_rep = gpu_S_vecs_2_rep.*gpu_choice_n_bar_rep;
%gpu_R_rep = gpu_S_vecs_1_rep.*gpu_choice_n_bar_rep_trans;

gpu_AmB_rep = arrayfun(@times,gpu_S_vecs_2_rep,gpu_choice_n_bar_rep);
gpu_R_rep = arrayfun(@times,gpu_S_vecs_1_rep,gpu_choice_n_bar_rep_trans);




gpu_K_rep = repmat(gpu_S_vecs_3,1,num_reps);

tic()
gpu_y_reps = gpuArray(y_reps);
gpu_S_rep = gpu_y_reps(1:n_max,:);
gpu_C_rep = gpu_y_reps(n_max+1:end,:);


%gpu_CCK_rep =  (gpu_C_rep./(gpu_C_rep + gpu_K_rep));

gpu_CCK_rep = arrayfun(@rdivide,gpu_C_rep,gpu_K_rep);

gpu_R_CCK_rep  = pagefun(@mtimes,gpu_R_rep,gpu_CCK_rep);

%gpu_dydt1 = (-gpu_delta + gpu_R_CCK_rep).*gpu_S_rep;

gpu_dydt1 = arrayfun(@times,(-gpu_delta + gpu_R_CCK_rep),gpu_S_rep);


gpu_AmB_S_rep = pagefun(@mtimes,gpu_AmB_rep,gpu_S_rep);

gpu_dydt2 = -gpu_delta*gpu_C_rep + gpu_AmB_S_rep;

gpu_dydt = [gpu_dydt1;gpu_dydt2];
toc()




%% CPU vectorized

choice_n_rep_aug = reshape(choice_n_rep,1,n_max,num_reps);
choice_n_reps_trans = permute(choice_n_rep_aug,[2,1,3]);

choice_n_bar_rep = repmat(  choice_n_rep_aug , m_max, 1, 1  );
choice_n_bar_rep_trans = repmat(  choice_n_reps_trans , 1, m_max, 1  );
AmB_rep = S_vecs_2_rep.*choice_n_bar_rep;

R_rep = S_vecs_1_rep.*choice_n_bar_rep_trans;

K_rep = repmat(S_vecs(:,1,3),1,num_reps);

tic()
S_rep = y_reps(1:n_max,:);
C_rep = y_reps(n_max+1:end,:);

CCK_rep =  (C_rep./(C_rep + K_rep));

R_CCK_rep  = cpumanymult(R_rep,CCK_rep);

dydt1 = (-delta + R_CCK_rep).*S_rep;

AmB_S_rep = cpumanymult(AmB_rep,S_rep);

dydt2 = -delta*C_rep + AmB_S_rep;

dydt = [dydt1;dydt2];
toc()


%% direct for-loop
tic()
dydt_direct = zeros(n_max + m_max,num_reps);
for i = 1:num_reps
    dydt_direct(:,i) = model3( y_reps(:,i) , S_vecs , choice_n_rep(:,i) ,delta );
end
toc()










% is doing .* with gpuArrays the same as applying gpufun?
% can I do .* with gpu arrays like I do on cpu with multiplicity copying?


%% solving a simple gradient computation on GPU

tic()
gpu_choice_n = gpuArray(choice_n);
gpu_delta = gpuArray(delta);
gpu_S_vecs = gpuArray(S_vecs);
gpu_y = gpuArray(y);
[gpu_dydt] = model3( gpu_y , gpu_S_vecs , gpu_choice_n ,gpu_delta );
cpu_gpu_dydt = gather(gpu_dydt);
toc()

tic()
[dydt] = model3( y , S_vecs , choice_n ,delta );
toc()

norm(cpu_gpu_dydt-dydt)
%% applying model3 to multiple matrices in parallel


pagefun




%% simple test to compare the speed of GPU on operating on a list of matrices

n = 100;
list_of_A = rand(n);
list_of_B = rand(n);

gpulist_of_A = gpuArray(list_of_A);
gpulist_of_B = gpuArray(list_of_B);


gpulist_of_C = arrayfun(@sum_and_product,gpulist_of_A,gpulist_of_B);

C = gather(gpulist_of_C);

%%


function c = sum_and_product(a,b)
n = 100;
oa = 0;
ob = 0;
for i = 1:n
oa = oa + a(i);
ob = ob + b(i);
end
c = oa*ob;
end

%%

function [dydt] = model3( y , S_vecs , choice_n ,delta )

m_max = size(S_vecs,1);
n_max = size(S_vecs,2);

S = y(1:n_max);
C = y(n_max+1:end);
K = S_vecs(:,1,3);
AmB = S_vecs(:,:,2).*(choice_n');
R = (S_vecs(:,:,1)').*choice_n;

dydt1 = (-delta + R*(C./(C + K))).*S;

dydt2 = -delta*C + (AmB*S);

dydt = [dydt1;dydt2];


end

function y = cpumanylinemult(A,v)

y = zeros(size(A,1),size(A,2),size(v,2));
for i = 1:size(v,2)
    y(:,:,i) = A.*(v(:,i)');
end

end




function y = cpumanymult(A,v)

y = zeros(size(A,1),size(v,2));
for i = 1:size(v,2)
    y(:,i) = A(:,:,i)*v(:,i);
end

end
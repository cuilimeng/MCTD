function [ U1, U2, P1, P2, Z, I, V ] = solveSTDH( X1, X2, L, para)
%% identify the parameters
alpha = para.alpha;
beta = para.beta;
gamma = para.gamma;
mu = para.mu;
lambda = para.lambda;
bits = para.bits;
class_num = para.class_num;
sub_dim = para.sub_dim;
n_c = cumsum(para.tr_n_c);

%% random initialization
[row1, col1] = size(X1);
[row2, ~] = size(X2);
V = rand(bits, col1);
Ada_V = zeros(size(V));
Z = rand(sub_dim*class_num,bits);
Ada_Z = zeros(size(Z));
U1 = rand(row1, bits);
U2 = rand(row2, bits);
P1 = rand(bits, row1);
P2 = rand(bits, row2);
I = ones(sub_dim*class_num, col1);
for i = 1:class_num
    I((i-1)*sub_dim+1:i*sub_dim,n_c(i)+1:n_c(i+1)) = 0;
end
threshold = 1;
lastF = 99999999;
iter = 1;

%% compute iteratively
while (true)
	%% update U1 and U2
    U1 = X1 * V' / (V * V' + (lambda / alpha) * eye(bits));
    U2 = X2 * V' / (V * V' + (lambda / (1-alpha)) * eye(bits));
    
    %% update V    
    % iterative optimizing V
    A = 2 * (alpha * U1' * U1 + (1 - alpha) * U2' * U2 + 2 * beta * eye(bits)+ lambda * eye(bits));
    B = gamma * (L + L');
    C = -2 * (alpha * U1' * X1 + (1 - alpha) * U2' * X2 + beta * (P1 * X1 + P2 * X2))...
        + mu * (Z'*(I.*(Z*V)));
    grad_V = A * V + V * B + C;
    Ada_V = Ada_V + power(grad_V,2);
    V = V - 1 * (grad_V)./( sqrt(Ada_V) + 1e-6 );    
%     % Optimizing V by solving the Sylvester equation
%     A = 2 * (alpha * U1' * U1 + (1 - alpha) * U2' * U2 + 2 * beta * eye(bits)+ lambda * eye(bits));
%     B = gamma * (L + L');
%     C = -2 * (alpha * U1' * X1 + (1 - alpha) * U2' * X2 + beta * (P1 * X1 + P2 * X2));
%        
%     for i = 1:class_num
%         A1 = A + mu * Z' * diag(I(:,n_c(i)+1)) * Z;
%         B1 = B(n_c(i)+1:n_c(i+1),n_c(i)+1:n_c(i+1));
%         C1 = C(:,n_c(i)+1:n_c(i+1));
%         tmp_V = lyap(A1,B1,C1);
%         V(:,n_c(i)+1:n_c(i+1)) = tmp_V;
%     end
   
    %% update Z
    grad_Z = mu * ((I .* (Z * V)) * V') + lambda * Z;
    Ada_Z = Ada_Z + power(grad_Z,2);
    Z = Z - 0.1 * (grad_Z)./( sqrt(Ada_Z) + 1e-6 );
    % Z = Z - 0.01 * (grad_Z);

    %% update P1 and P2
    P1 = V * X1' / (X1 * X1' + (lambda / beta) * eye(row1));
    P2 = V * X2' / (X2 * X2' + (lambda / beta) * eye(row2));
  
    %% compute objective function
    norm1 = alpha * (norm(X1 - U1 * V, 'fro')^2);
    norm2 = (1 - alpha) * (norm(X2 - U2 * V, 'fro')^2);
    norm3 = beta * (norm(V - P1 * X1, 'fro')^2);
    norm4 = beta * (norm(V - P2 * X2, 'fro')^2);
    norm5 = gamma * trace(V * L * V');
    norm6 = lambda * (norm(U1, 'fro')^2 + norm(U2, 'fro')^2 + norm(V, 'fro')^2 + norm(P1, 'fro')^2 + norm(P2, 'fro')^2);
    norm7 = mu * (norm(I.* (Z * V), 'fro')^2);
    currentF= norm1 + norm2 + norm3 + norm4 + norm5 + norm6+ norm7;
    fprintf('\nobj at iteration %d: %.4f\n reconstruction error for structured tensor decomposition: %.4f,\n reconstruction error for linear projection: %.4f,\n diagonal constrain term: %.4f,\n joint graph term: %.4f,\n regularization term: %.4f\n\n', iter, currentF, norm1 + norm2, norm3 + norm4, norm7, norm5, norm6);
    if (lastF - currentF) < threshold
        fprintf('algorithm converges...\n');
        fprintf('final obj: %.4f\n reconstruction error for structured tensor decomposition: %.4f,\n reconstruction error for linear projection: %.4f,\n diagonal constrain term: %.4f,\n joint graph term: %.4f,\n regularization term: %.4f\n\n', currentF,norm1 + norm2, norm3 + norm4, norm7, norm5, norm6);
        return;
    end
    
    iter = iter + 1;
    lastF = currentF;
end
end


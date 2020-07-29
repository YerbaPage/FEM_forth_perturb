clear all
node = [0,0; 1,0; 1,1; 0,1];
elem = [2,3,1; 4,1,3];      
for k = 1:3
  [node,elem] = uniformrefine(node,elem);
end
% Homogenous Dirichlet boundary condition
pde.f = inline('ones(size(p,1),1)','p');
pde.g_D = 0;
u = Poisson(node,elem,[],pde);
figure(1); 
showresult(node,elem,u);